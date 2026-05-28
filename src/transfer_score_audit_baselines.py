from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from kbs_frontier_smoke_tests import calibrate, read_split, split_fit_calibration
from kbs_sds_kb_strengthening import (
    YEAR_PANEL_DIR,
    add_prototype_features_custom,
    augment_training,
    available,
    read_year_panel,
)
from smoke_fepl_topic import CATEGORICAL_FEATURES, NUMERIC_FEATURES, sample_train
from sds_kb_evidence_quality import fit_full_sds_kb


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "transfer_score_audit_baselines"


def top_mask(score: np.ndarray, share: float = 0.10) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(share * len(score))))
    idx = np.argpartition(-score, k - 1)[:k] if k < len(score) else np.arange(len(score))
    mask = np.zeros(len(score), dtype=bool)
    mask[idx] = True
    return mask


def metric_row(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prob_arr = np.clip(score if prob is None else np.asarray(prob, dtype=float), 0.0, 1.0)
    top = top_mask(score, 0.10)
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "top10_capture": float(y[top].sum() / max(y.sum(), 1)),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, prob_arr)),
    }


def make_ohe(
    fit_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    numeric_cols = [col for col in NUMERIC_FEATURES if col in fit_df.columns and col in test_df.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in fit_df.columns and col in test_df.columns]
    fit_num = sparse.csr_matrix(fit_df[numeric_cols].fillna(0.0).to_numpy(np.float32))
    cal_num = sparse.csr_matrix(cal_df[numeric_cols].fillna(0.0).to_numpy(np.float32))
    test_num = sparse.csr_matrix(test_df[numeric_cols].fillna(0.0).to_numpy(np.float32))
    if not categorical_cols:
        return fit_num, cal_num, test_num
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    fit_cat = enc.fit_transform(fit_df[categorical_cols].astype(str))
    cal_cat = enc.transform(cal_df[categorical_cols].astype(str))
    test_cat = enc.transform(test_df[categorical_cols].astype(str))
    return (
        sparse.hstack([fit_num, fit_cat], format="csr"),
        sparse.hstack([cal_num, cal_cat], format="csr"),
        sparse.hstack([test_num, test_cat], format="csr"),
    )


def fit_score_reference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    seed: int,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_sample = sample_train(train, max_rows, seed).reset_index(drop=True)
    fit_df, cal_df = split_fit_calibration(train_sample, seed, max_rows)
    y_fit = fit_df["severe_late_aircraft"].to_numpy(int)
    y_cal = cal_df["severe_late_aircraft"].to_numpy(int)
    scale_pos = max(float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)), 1.0)

    if model_name == "LightGBM":
        x_fit, x_cal, x_test = make_ohe(fit_df, cal_df, test)
        model = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(x_fit, y_fit)
        cal_score = model.predict_proba(x_cal)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]
    elif model_name == "XGBoost":
        x_fit, x_cal, x_test = make_ohe(fit_df, cal_df, test)
        model = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
        )
        model.fit(x_fit, y_fit)
        cal_score = model.predict_proba(x_cal)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]
    elif model_name == "CatBoost":
        numeric_cols = [col for col in NUMERIC_FEATURES if col in fit_df.columns and col in test.columns]
        cat_cols = [col for col in CATEGORICAL_FEATURES if col in fit_df.columns and col in test.columns]
        model_cols = numeric_cols + cat_cols
        cat_features = [model_cols.index(col) for col in cat_cols]
        model = CatBoostClassifier(
            iterations=240,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            random_seed=seed,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
        )
        x_fit = fit_df[model_cols].copy()
        x_cal = cal_df[model_cols].copy()
        x_test = test[model_cols].copy()
        model.fit(Pool(x_fit, y_fit, cat_features=cat_features))
        cal_score = model.predict_proba(Pool(x_cal, cat_features=cat_features))[:, 1]
        test_score = model.predict_proba(Pool(x_test, cat_features=cat_features))[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    test_prob = calibrate(cal_score, y_cal, test_score)
    return test_score, test_prob


def fit_pe_kb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    max_rows: int,
    fast: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not fast:
        _, test_kb, info, score, prob = fit_full_sds_kb(train, test, seed, max_rows)
        if info is None:
            raise RuntimeError("PE-KB evidence requires nearest-prototype information.")
        return test_kb, score, prob

    train_kb, test_kb, kb_cols, _ = add_prototype_features_custom(
        train,
        test,
        seed=seed,
        max_proto=3000,
        neighbors=5,
        use_fuzzy=True,
        return_neighbors=False,
    )
    raw_cols = available(train_kb, [
        "prev_arr_delay",
        "prev_late_aircraft_delay",
        "scheduled_turnaround",
        "actual_turnaround",
        "turnaround_slack",
        "recovery_shortfall",
        "crs_elapsed",
        "actual_elapsed",
        "distance",
        "dep_hour",
        "origin_hour_chain_density",
    ])
    fuzzy_cols = available(train_kb, [
        "mu_upstream_delay",
        "mu_prev_late_aircraft",
        "mu_tight_turnaround",
        "mu_recovery_shortfall",
        "mu_origin_chain_density",
        "mu_buffer_failure_path",
        "mu_late_recurrence_path",
        "mu_congested_carryover_path",
    ])
    cols = [*raw_cols, *fuzzy_cols, *kb_cols]
    fit_df, cal_df = split_fit_calibration(train_kb, seed, max_rows)
    fit_aug = augment_training(fit_df, cols, seed, 1.0)
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=seed,
        class_weight="balanced",
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(fit_aug[cols].fillna(0.0), fit_aug["severe_late_aircraft"].to_numpy(int))
    cal_score = model.predict_proba(cal_df[cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test_kb[cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
    return test_kb, test_score, test_prob


def queue_stats(
    protocol: str,
    fold: str,
    queue: str,
    y: np.ndarray,
    score: np.ndarray,
    prob: np.ndarray,
    margin: np.ndarray,
) -> dict[str, float | int | str]:
    top = top_mask(score, 0.10)
    severe = y == 1
    captured_mask = top & severe
    supported_mask = captured_mask & (margin > 0.0)
    reviewed = int(top.sum())
    captured = int(captured_mask.sum())
    supported = int(supported_mask.sum())
    row: dict[str, float | int | str] = {
        "protocol": protocol,
        "fold": fold,
        "queue": queue,
        **metric_row(y, score, prob),
        "reviewed_paths": reviewed,
        "captured_severe_cases": captured,
        "evidence_supported_severe_cases": supported,
        "evidence_hit_rate": float(supported / max(captured, 1)),
        "severe_cases_per_100_reviews": float(100.0 * captured / max(reviewed, 1)),
        "evidence_supported_cases_per_100_reviews": float(100.0 * supported / max(reviewed, 1)),
    }
    return row


def evaluate_fold(
    protocol: str,
    fold: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: list[str],
    seed: int,
    max_rows: int,
    fast_pe_kb: bool,
) -> list[dict[str, float | int | str]]:
    if train.empty or test.empty:
        return []
    test_kb, pe_score, pe_prob = fit_pe_kb(train, test, seed, max_rows, fast=fast_pe_kb)
    y = test_kb["severe_late_aircraft"].to_numpy(int)
    margin = test_kb["kb_distance_margin"].to_numpy(float)
    rows = [queue_stats(protocol, fold, "PE-KB queue", y, pe_score, pe_prob, margin)]
    for model_name in models:
        score, prob = fit_score_reference(train, test_kb, model_name, seed, max_rows)
        rows.append(queue_stats(protocol, fold, f"{model_name} score + PE-KB evidence", y, score, prob, margin))
    return rows


def airport_folds(splits: list[str]) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    folds = []
    for split in splits:
        train, test, _ = read_split(split)
        heldout = set(test["origin"].value_counts().head(20).index)
        folds.append((split, train[~train["origin"].isin(heldout)].copy(), test[test["origin"].isin(heldout)].copy()))
    return folds


def carrier_folds(splits: list[str], max_carriers: int | None = None) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    folds = []
    for split in splits:
        train, test, _ = read_split(split)
        carriers = sorted(set(train["carrier"]).intersection(set(test["carrier"])))
        if max_carriers is not None:
            carriers = carriers[:max_carriers]
        for carrier in carriers:
            folds.append(
                (
                    f"{split}_{carrier}",
                    train[train["carrier"].ne(carrier)].copy(),
                    test[test["carrier"].eq(carrier)].copy(),
                )
            )
    return folds


def cross_year_folds() -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    train = read_year_panel(YEAR_PANEL_DIR / "fepl_paths_2024_all_AA-DL-OO-UA-WN_yearonly.csv")
    test = read_year_panel(YEAR_PANEL_DIR / "fepl_paths_2025_all_AA-DL-OO-UA-WN_prevdec.csv")
    return [("train2024_test2025", train, test)]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, queue), g in detail.groupby(["protocol", "queue"], sort=False):
        reviewed = int(g["reviewed_paths"].sum())
        captured = int(g["captured_severe_cases"].sum())
        supported = int(g["evidence_supported_severe_cases"].sum())
        row: dict[str, float | int | str] = {
            "protocol": protocol,
            "queue": queue,
            "folds": int(g["fold"].nunique()),
            "rows": int(g["rows"].sum()),
            "positives": int(g["positives"].sum()),
            "reviewed_paths": reviewed,
            "captured_severe_cases": captured,
            "evidence_supported_severe_cases": supported,
            "pooled_evidence_hit_rate": float(supported / max(captured, 1)),
            "pooled_severe_cases_per_100_reviews": float(100.0 * captured / max(reviewed, 1)),
            "pooled_evidence_supported_cases_per_100_reviews": float(100.0 * supported / max(reviewed, 1)),
        }
        for metric in ["top10_capture", "auc", "pr_auc", "brier", "evidence_hit_rate"]:
            vals = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Transfer score-compatible audit baselines",
        "",
        "Each protocol first forms a score queue. PE-KB prototype evidence is then attached to the queued paths through the fixed path-event evidence channel.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold detail",
        "",
        detail.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out / "transfer_score_audit_baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--protocols", default="airport")
    parser.add_argument("--splits", default="2025_01_to_2025_02")
    parser.add_argument("--models", default="LightGBM")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--max-carriers", type=int, default=None)
    parser.add_argument("--fast-pe-kb", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    protocols = parse_list(args.protocols)
    splits = parse_list(args.splits)
    models = parse_list(args.models)
    all_rows: list[dict[str, float | int | str]] = []

    for protocol in protocols:
        if protocol == "airport":
            folds = airport_folds(splits)
        elif protocol == "carrier":
            folds = carrier_folds(splits, max_carriers=args.max_carriers)
        elif protocol == "cross_year":
            folds = cross_year_folds()
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
        for fold, train, test in folds:
            all_rows.extend(
                evaluate_fold(
                    protocol,
                    fold,
                    train,
                    test,
                    models=models,
                    seed=args.seed,
                    max_rows=args.max_train_rows,
                    fast_pe_kb=args.fast_pe_kb,
                )
            )

    detail = pd.DataFrame(all_rows)
    summary = summarize(detail)
    detail.to_csv(args.out / "transfer_score_audit_baselines_detail.csv", index=False)
    summary.to_csv(args.out / "transfer_score_audit_baselines_summary.csv", index=False)
    write_report(args.out, detail, summary)
    print((args.out / "transfer_score_audit_baselines_report.md").resolve())


if __name__ == "__main__":
    main()

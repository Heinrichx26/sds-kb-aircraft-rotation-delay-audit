from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kbs_frontier_smoke_tests import calibrate, fit_lgbm, read_split, split_fit_calibration
from kbs_sds_kb_strengthening import (
    FUZZY_COLS,
    RAW_COLS,
    add_prototype_features_custom,
    augment_training,
    available,
    metric_dict,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "sds_kb_evidence_side_ablation"


VARIANTS = [
    {
        "variant": "Full SDS-KB",
        "score_prototypes": True,
        "score_fuzzy": True,
        "score_simulation": True,
        "evidence_fuzzy": True,
    },
    {
        "variant": "No simulated severe store",
        "score_prototypes": True,
        "score_fuzzy": True,
        "score_simulation": False,
        "evidence_fuzzy": True,
    },
    {
        "variant": "No prototype affinity in scorer",
        "score_prototypes": False,
        "score_fuzzy": True,
        "score_simulation": True,
        "evidence_fuzzy": True,
    },
    {
        "variant": "No fuzzy concepts",
        "score_prototypes": True,
        "score_fuzzy": False,
        "score_simulation": True,
        "evidence_fuzzy": False,
    },
]


def top_share_mask(score: np.ndarray, share: float = 0.10) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(share * len(score))))
    order = np.argpartition(-score, k - 1)[:k] if k < len(score) else np.arange(len(score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order] = True
    return mask


def fit_variant_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
    seed: int,
    max_rows: int,
    max_proto: int,
    neighbors: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_work = train.copy()
    test_work = test.copy()
    kb_cols: list[str] = []
    if cfg["score_prototypes"]:
        train_work, test_work, kb_cols, _ = add_prototype_features_custom(
            train_work,
            test_work,
            seed=seed,
            max_proto=max_proto,
            neighbors=neighbors,
            use_fuzzy=cfg["score_fuzzy"],
            return_neighbors=False,
        )
    raw_cols = available(train_work, RAW_COLS)
    fuzzy_cols = available(train_work, FUZZY_COLS) if cfg["score_fuzzy"] else []
    feature_cols = [*raw_cols, *fuzzy_cols, *kb_cols]
    fit_df, cal_df = split_fit_calibration(train_work, seed, max_rows)
    if cfg["score_simulation"]:
        fit_df = augment_training(fit_df, feature_cols, seed, 1.0)
    model = fit_lgbm(
        fit_df[feature_cols].fillna(0.0),
        fit_df["severe_late_aircraft"].to_numpy(int),
        seed,
        n_estimators=n_estimators,
    )
    cal_score = model.predict_proba(cal_df[feature_cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test_work[feature_cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
    scoring = metric_dict(test_work["severe_late_aircraft"].to_numpy(int), test_score, test_prob)
    return test_score, test_prob, scoring


def attach_evidence(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    max_proto: int,
    neighbors: int,
    use_fuzzy: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    _, test_kb, _, info = add_prototype_features_custom(
        train,
        test,
        seed=seed,
        max_proto=max_proto,
        neighbors=neighbors,
        use_fuzzy=use_fuzzy,
        return_neighbors=True,
    )
    if info is None:
        raise RuntimeError("Evidence-side ablation requires prototype neighbor information.")
    return test_kb, info


def evidence_metrics(
    split: str,
    variant: str,
    train: pd.DataFrame,
    test_kb: pd.DataFrame,
    info: dict[str, np.ndarray],
    score: np.ndarray,
    prob: np.ndarray,
    scoring: dict[str, float],
) -> dict[str, float | str]:
    y = test_kb["severe_late_aircraft"].to_numpy(int)
    margin = test_kb["kb_distance_margin"].to_numpy(float)
    top_mask = top_share_mask(score, 0.10)
    severe_mask = y == 1
    normal_mask = y == 0
    top_severe = top_mask & severe_mask
    reviewed = int(top_mask.sum())
    captured = int(top_severe.sum())
    supported_mask = top_severe & (margin > 0.0)
    supported = int(supported_mask.sum())
    nearest_severe = train.iloc[info["test_nearest_severe_train_idx"]].reset_index(drop=True)
    nearest_normal = train.iloc[info["test_nearest_normal_train_idx"]].reset_index(drop=True)
    nearest_severe_delay = nearest_severe["LateAircraftDelay"].to_numpy(float)
    nearest_normal_delay = nearest_normal["LateAircraftDelay"].to_numpy(float)
    nearest_ids = info["test_nearest_severe_train_idx"]
    unique_supported = int(len(np.unique(nearest_ids[supported_mask]))) if supported else 0
    return {
        "split": split,
        "variant": variant,
        "rows": int(len(test_kb)),
        "positives": int(y.sum()),
        "reviewed_paths": reviewed,
        "captured_severe_cases": captured,
        "evidence_supported_severe_cases": supported,
        "evidence_hit_rate": float(supported / captured) if captured else np.nan,
        "evidence_supported_cases_per_100_reviews": float(100.0 * supported / reviewed) if reviewed else np.nan,
        "severe_cases_per_100_reviews": float(100.0 * captured / reviewed) if reviewed else np.nan,
        "margin_gap": float(np.nanmedian(margin[severe_mask]) - np.nanmedian(margin[normal_mask])),
        "top10_margin_lift": float(
            np.nanmean(margin[top_severe]) - np.nanmean(margin[top_mask & normal_mask])
        )
        if top_severe.any() and (top_mask & normal_mask).any()
        else np.nan,
        "prototype_delay_gap": float(
            np.nanmedian(nearest_severe_delay[top_severe]) - np.nanmedian(nearest_normal_delay[top_severe])
        )
        if top_severe.any()
        else np.nan,
        "unique_supported_severe_prototypes": unique_supported,
        "unique_supported_prototypes_per_100_supported": float(100.0 * unique_supported / supported) if supported else np.nan,
        "score_top10_capture": scoring["top10_capture"],
        "score_auc": scoring["auc"],
        "score_pr_auc": scoring["pr_auc"],
        "score_brier": scoring["brier"],
    }


def run_split(split: str, seed: int, max_rows: int, max_proto: int, neighbors: int, n_estimators: int) -> pd.DataFrame:
    train, test, _ = read_split(split)
    rows = []
    evidence_cache: dict[bool, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    for cfg in VARIANTS:
        score, prob, scoring = fit_variant_score(
            train,
            test,
            cfg,
            seed=seed,
            max_rows=max_rows,
            max_proto=max_proto,
            neighbors=neighbors,
            n_estimators=n_estimators,
        )
        key = bool(cfg["evidence_fuzzy"])
        if key not in evidence_cache:
            evidence_cache[key] = attach_evidence(train, test, seed, max_proto, neighbors, use_fuzzy=key)
        test_kb, info = evidence_cache[key]
        rows.append(evidence_metrics(split, cfg["variant"], train, test_kb, info, score, prob, scoring))
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in detail.groupby("variant", sort=False):
        reviewed = int(g["reviewed_paths"].sum())
        captured = int(g["captured_severe_cases"].sum())
        supported = int(g["evidence_supported_severe_cases"].sum())
        row: dict[str, float | str | int] = {
            "variant": variant,
            "splits": int(g["split"].nunique()),
            "reviewed_paths": reviewed,
            "captured_severe_cases": captured,
            "evidence_supported_severe_cases": supported,
            "pooled_evidence_hit_rate": float(supported / captured) if captured else np.nan,
            "pooled_evidence_supported_cases_per_100_reviews": float(100.0 * supported / reviewed) if reviewed else np.nan,
        }
        for col in [
            "evidence_hit_rate",
            "margin_gap",
            "top10_margin_lift",
            "prototype_delay_gap",
            "unique_supported_prototypes_per_100_supported",
            "score_pr_auc",
            "score_brier",
        ]:
            vals = g[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    cols = [
        "variant",
        "pooled_evidence_hit_rate",
        "margin_gap_mean",
        "margin_gap_std",
        "prototype_delay_gap_mean",
        "pooled_evidence_supported_cases_per_100_reviews",
        "unique_supported_prototypes_per_100_supported_mean",
        "score_pr_auc_mean",
    ]
    lines = [
        "# SDS-KB evidence-side ablation",
        "",
        "This diagnostic keeps the ablation score queues and then evaluates the prototype-evidence attached to those queues.",
        "The no-prototype-affinity row removes prototype features from the scorer and then attaches the same fixed evidence channel after the queue is formed.",
        "The no-fuzzy-concepts row computes prototype evidence from recorded path quantities without fuzzy concept memberships.",
        "",
        "## Summary",
        "",
        summary[cols].to_markdown(index=False),
        "",
        "## Split-level detail",
        "",
        detail.to_markdown(index=False),
        "",
    ]
    (out / "sds_kb_evidence_side_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="2025_07_to_2025_08")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--max-proto", type=int, default=4500)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=280)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    frames = []
    for split in splits:
        split_df = run_split(split, args.seed, args.max_train_rows, args.max_proto, args.neighbors, args.n_estimators)
        frames.append(split_df)
        print(split_df[["split", "variant", "evidence_hit_rate", "margin_gap", "prototype_delay_gap", "evidence_supported_cases_per_100_reviews"]].to_string(index=False))
    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    detail.to_csv(args.out / "sds_kb_evidence_side_ablation_detail.csv", index=False)
    summary.to_csv(args.out / "sds_kb_evidence_side_ablation_summary.csv", index=False)
    write_report(args.out, detail, summary)
    print("Summary")
    print(summary[["variant", "pooled_evidence_hit_rate", "margin_gap_mean", "prototype_delay_gap_mean", "pooled_evidence_supported_cases_per_100_reviews", "score_pr_auc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()

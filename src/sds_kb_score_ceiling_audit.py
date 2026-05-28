from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from kbs_frontier_smoke_tests import read_split
from sds_kb_evidence_quality import fit_full_sds_kb
from smoke_fepl_topic import CATEGORICAL_FEATURES, NUMERIC_FEATURES, sample_train


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "sds_kb_score_ceiling_audit"


def top_mask(score: np.ndarray, share: float = 0.10) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(share * len(score))))
    idx = np.argpartition(-score, k - 1)[:k] if k < len(score) else np.arange(len(score))
    mask = np.zeros(len(score), dtype=bool)
    mask[idx] = True
    return mask


def fit_lightgbm_score(train: pd.DataFrame, test: pd.DataFrame, seed: int, max_train_rows: int) -> np.ndarray:
    train_fit = sample_train(train, max_train_rows, seed)
    y = train_fit["severe_late_aircraft"].to_numpy(int)
    x_train = pd.concat(
        [
            train_fit[NUMERIC_FEATURES].astype(float),
            pd.get_dummies(train_fit[CATEGORICAL_FEATURES].astype(str), dtype=float),
        ],
        axis=1,
    )
    x_test = pd.concat(
        [
            test[NUMERIC_FEATURES].astype(float),
            pd.get_dummies(test[CATEGORICAL_FEATURES].astype(str), dtype=float),
        ],
        axis=1,
    ).reindex(columns=x_train.columns, fill_value=0.0)
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
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1]


def metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prob_arr = np.clip(score if prob is None else np.asarray(prob, dtype=float), 0.0, 1.0)
    mask = top_mask(score, 0.10)
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "top10_capture": float(y[mask].sum() / max(y.sum(), 1)),
        "top10_hit_rate": float(y[mask].mean()),
        "auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, prob_arr)),
    }


def evaluate_split(split: str, seed: int, max_train_rows: int) -> dict[str, float | str]:
    train, test, _ = read_split(split)
    _, test_kb, _, sds_score, sds_prob = fit_full_sds_kb(train, test, seed, max_train_rows)
    lgbm_score = fit_lightgbm_score(train, test, seed, max_train_rows)
    y = test_kb["severe_late_aircraft"].to_numpy(int)
    margin = test_kb["kb_distance_margin"].to_numpy(float)

    sds_top = top_mask(sds_score, 0.10)
    lgbm_top = top_mask(lgbm_score, 0.10)
    severe = y == 1

    def queue_stats(name: str, mask: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | str]:
        captured = mask & severe
        supported = captured & (margin > 0.0)
        normal_in_queue = mask & (~severe)
        row = {
            "split": split,
            "queue": name,
            **metrics(y, score, prob),
            "reviewed_paths": int(mask.sum()),
            "captured_severe_cases": int(captured.sum()),
            "evidence_supported_severe_cases": int(supported.sum()),
            "evidence_hit_rate": float(supported.sum() / max(captured.sum(), 1)),
            "severe_cases_per_100_reviews": float(100.0 * captured.sum() / max(mask.sum(), 1)),
            "evidence_supported_cases_per_100_reviews": float(100.0 * supported.sum() / max(mask.sum(), 1)),
            "top10_severe_margin_median": float(np.nanmedian(margin[captured])) if captured.any() else np.nan,
            "top10_normal_margin_median": float(np.nanmedian(margin[normal_in_queue])) if normal_in_queue.any() else np.nan,
        }
        return row

    rows = [
        queue_stats("SDS-KB queue", sds_top, sds_score, sds_prob),
        queue_stats("LightGBM score queue with SDS-KB evidence", lgbm_top, lgbm_score, lgbm_score),
    ]
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for queue, g in detail.groupby("queue", sort=False):
        reviewed = int(g["reviewed_paths"].sum())
        captured = int(g["captured_severe_cases"].sum())
        supported = int(g["evidence_supported_severe_cases"].sum())
        row = {
            "queue": queue,
            "splits": int(g["split"].nunique()),
            "rows": int(g["rows"].sum()),
            "positives": int(g["positives"].sum()),
            "reviewed_paths": reviewed,
            "captured_severe_cases": captured,
            "evidence_supported_severe_cases": supported,
            "aggregate_evidence_hit_rate": float(supported / max(captured, 1)),
            "aggregate_severe_cases_per_100_reviews": float(100.0 * captured / max(reviewed, 1)),
            "aggregate_evidence_supported_cases_per_100_reviews": float(100.0 * supported / max(reviewed, 1)),
        }
        for metric in [
            "top10_capture",
            "auc",
            "pr_auc",
            "brier",
            "evidence_hit_rate",
            "top10_severe_margin_median",
            "top10_normal_margin_median",
        ]:
            row[metric] = float(np.nanmean(g[metric].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# SDS-KB score-ceiling audit bridge",
        "",
        "This diagnostic keeps score ranking and audit evidence separate. The LightGBM queue is formed only from the score-ceiling model, and SDS-KB supplies fixed prototype evidence after that queue is fixed.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split detail",
        "",
        detail.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out / "score_ceiling_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--splits", default="2025_01_to_2025_02")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    split_list = [item.strip() for item in args.splits.split(",") if item.strip()]
    all_rows = []
    for split in split_list:
        all_rows.extend(evaluate_split(split, args.seed, args.max_train_rows))
    detail = pd.DataFrame(all_rows)
    summary = summarize(detail)
    detail.to_csv(args.out / "score_ceiling_audit_detail.csv", index=False)
    summary.to_csv(args.out / "score_ceiling_audit_summary.csv", index=False)
    write_report(args.out, detail, summary)
    print((args.out / "score_ceiling_audit_report.md").resolve())


if __name__ == "__main__":
    main()

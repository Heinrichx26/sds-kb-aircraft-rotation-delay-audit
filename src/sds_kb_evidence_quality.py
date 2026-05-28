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
    active_concepts,
    add_prototype_features_custom,
    augment_training,
    available,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "sds_kb_evidence_quality"

CONCEPT_LABELS = {
    "mu_buffer_failure_path": "Buffer",
    "mu_late_recurrence_path": "Recurrence",
    "mu_congested_carryover_path": "Congested",
    "mu_upstream_delay": "Upstream",
    "mu_recovery_shortfall": "Shortfall",
    "mu_tight_turnaround": "Tight turn",
}


def percentile_dict(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {f"{prefix}_p25": np.nan, f"{prefix}_p50": np.nan, f"{prefix}_p75": np.nan}
    return {
        f"{prefix}_p25": float(np.nanpercentile(values, 25)),
        f"{prefix}_p50": float(np.nanpercentile(values, 50)),
        f"{prefix}_p75": float(np.nanpercentile(values, 75)),
    }


def top_share_mask(score: np.ndarray, share: float = 0.10) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(share * len(score))))
    order = np.argpartition(-score, k - 1)[:k] if k < len(score) else np.arange(len(score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order] = True
    return mask


def concept_set(row: pd.Series, threshold: float = 0.5) -> set[str]:
    active = set()
    for col, label in CONCEPT_LABELS.items():
        if float(row.get(col, 0.0)) >= threshold:
            active.add(label)
    return active


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def fit_full_sds_kb(train: pd.DataFrame, test: pd.DataFrame, seed: int, max_rows: int):
    train_kb, test_kb, kb_cols, info = add_prototype_features_custom(
        train,
        test,
        seed=seed,
        max_proto=4500,
        neighbors=5,
        use_fuzzy=True,
        return_neighbors=True,
    )
    raw_cols = available(train_kb, RAW_COLS)
    fuzzy_cols = available(train_kb, FUZZY_COLS)
    cols = [*raw_cols, *fuzzy_cols, *kb_cols]
    fit_df, cal_df = split_fit_calibration(train_kb, seed, max_rows)
    fit_aug = augment_training(fit_df, cols, seed, 1.0)
    model = fit_lgbm(
        fit_aug[cols].fillna(0.0),
        fit_aug["severe_late_aircraft"].to_numpy(int),
        seed,
        n_estimators=280,
    )
    cal_score = model.predict_proba(cal_df[cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test_kb[cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
    return train_kb, test_kb, info, test_score, test_prob


def reliability_rows(split: str, model: str, prob: np.ndarray, y: np.ndarray, bins: int = 15) -> list[dict]:
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=int)
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for b in range(bins):
        low = edges[b]
        high = edges[b + 1]
        if b == bins - 1:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if not mask.any():
            continue
        rows.append(
            {
                "split": split,
                "model": model,
                "bin": b,
                "bin_low": low,
                "bin_high": high,
                "rows": int(mask.sum()),
                "mean_pred": float(np.mean(prob[mask])),
                "observed_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def evaluate_split(split: str, seed: int, max_rows: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, _ = read_split(split)
    train_kb, test_kb, info, score, prob = fit_full_sds_kb(train, test, seed, max_rows)
    if info is None:
        raise RuntimeError("Prototype neighbor information is required for evidence quality.")
    test_kb = test_kb.copy()
    test_kb["sds_kb_score"] = score
    test_kb["sds_kb_prob"] = prob
    y = test_kb["severe_late_aircraft"].to_numpy(int)
    margin = test_kb["kb_distance_margin"].to_numpy(float)
    top_mask = top_share_mask(score, 0.10)
    severe_mask = y == 1
    normal_mask = y == 0
    top_severe = top_mask & severe_mask
    top_normal = top_mask & normal_mask

    nearest_severe = train.iloc[info["test_nearest_severe_train_idx"]].reset_index(drop=True)
    nearest_normal = train.iloc[info["test_nearest_normal_train_idx"]].reset_index(drop=True)
    nearest_severe_delay = nearest_severe["LateAircraftDelay"].to_numpy(float)
    nearest_normal_delay = nearest_normal["LateAircraftDelay"].to_numpy(float)
    nearest_severe_shortfall = nearest_severe["recovery_shortfall"].to_numpy(float)
    nearest_normal_shortfall = nearest_normal["recovery_shortfall"].to_numpy(float)

    active_sets = [concept_set(row) for _, row in test_kb.iterrows()]
    top_severe_sets = [active_sets[i] for i in np.where(top_severe)[0]]
    top_concept_union = set().union(*top_severe_sets) if top_severe_sets else set()

    row = {
        "split": split,
        "rows": int(len(test_kb)),
        "positives": int(y.sum()),
        "top10_rows": int(top_mask.sum()),
        "top10_severe_cases": int(top_severe.sum()),
        "severe_margin_mean": float(np.nanmean(margin[severe_mask])),
        "normal_margin_mean": float(np.nanmean(margin[normal_mask])),
        "severe_margin_median": float(np.nanmedian(margin[severe_mask])),
        "normal_margin_median": float(np.nanmedian(margin[normal_mask])),
        "separation_mean_gap": float(np.nanmean(margin[severe_mask]) - np.nanmean(margin[normal_mask])),
        "separation_median_gap": float(np.nanmedian(margin[severe_mask]) - np.nanmedian(margin[normal_mask])),
        "top10_severe_margin_mean": float(np.nanmean(margin[top_severe])) if top_severe.any() else np.nan,
        "top10_normal_margin_mean": float(np.nanmean(margin[top_normal])) if top_normal.any() else np.nan,
        "top10_margin_lift": float(np.nanmean(margin[top_severe]) - np.nanmean(margin[top_normal])) if top_severe.any() and top_normal.any() else np.nan,
        "evidence_hit_rate": float(np.mean(margin[top_severe] > 0.0)) if top_severe.any() else np.nan,
        "nearest_severe_delay_median_top10_severe": float(np.nanmedian(nearest_severe_delay[top_severe])) if top_severe.any() else np.nan,
        "nearest_normal_delay_median_top10_severe": float(np.nanmedian(nearest_normal_delay[top_severe])) if top_severe.any() else np.nan,
        "prototype_delay_gap_top10_severe": float(np.nanmedian(nearest_severe_delay[top_severe]) - np.nanmedian(nearest_normal_delay[top_severe])) if top_severe.any() else np.nan,
        "nearest_severe_shortfall_median_top10_severe": float(np.nanmedian(nearest_severe_shortfall[top_severe])) if top_severe.any() else np.nan,
        "nearest_normal_shortfall_median_top10_severe": float(np.nanmedian(nearest_normal_shortfall[top_severe])) if top_severe.any() else np.nan,
        "top_evidence_concepts": "; ".join(sorted(top_concept_union)),
    }
    row.update(percentile_dict(margin[severe_mask], "severe_margin"))
    row.update(percentile_dict(margin[normal_mask], "normal_margin"))

    concept_rows = []
    for col, label in CONCEPT_LABELS.items():
        vals = test_kb.loc[top_severe, col].to_numpy(float) if top_severe.any() else np.asarray([])
        concept_rows.append(
            {
                "split": split,
                "concept": label,
                "top10_severe_active_rate": float(np.mean(vals >= 0.5)) if len(vals) else np.nan,
                "top10_severe_mean_membership": float(np.nanmean(vals)) if len(vals) else np.nan,
            }
        )

    order = np.argsort(-score)
    case_rows = []
    selected = [idx for idx in order if int(test_kb.iloc[idx]["severe_late_aircraft"]) == 1][:5]
    for rank, idx in enumerate(selected, start=1):
        path = test_kb.iloc[idx]
        sev_proto = nearest_severe.iloc[idx]
        norm_proto = nearest_normal.iloc[idx]
        case_rows.append(
            {
                "case": f"{split}_case{rank}",
                "split": split,
                "carrier": path.get("carrier"),
                "path": f"{path.get('prev_origin')}->{path.get('origin')}->{path.get('dest')}",
                "delay": float(path.get("LateAircraftDelay", np.nan)),
                "risk": float(path.get("sds_kb_prob", np.nan)),
                "severe_distance": float(path.get("kb_dist_severe", np.nan)),
                "normal_distance": float(path.get("kb_dist_normal", np.nan)),
                "distance_margin": float(path.get("kb_distance_margin", np.nan)),
                "active_concepts": active_concepts(path),
                "nearest_severe_path": f"{sev_proto.get('prev_origin')}->{sev_proto.get('origin')}->{sev_proto.get('dest')}",
                "nearest_severe_delay": float(sev_proto.get("LateAircraftDelay", np.nan)),
                "nearest_normal_path": f"{norm_proto.get('prev_origin')}->{norm_proto.get('origin')}->{norm_proto.get('dest')}",
                "nearest_normal_delay": float(norm_proto.get("LateAircraftDelay", np.nan)),
            }
        )
    rel = pd.DataFrame(reliability_rows(split, "SDS-KB", prob, y))
    return row, pd.DataFrame(concept_rows), pd.DataFrame(case_rows), rel


def summarize(evidence: pd.DataFrame, concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "separation_mean_gap",
        "separation_median_gap",
        "top10_margin_lift",
        "evidence_hit_rate",
        "nearest_severe_delay_median_top10_severe",
        "nearest_normal_delay_median_top10_severe",
        "prototype_delay_gap_top10_severe",
    ]
    rows = []
    for col in metric_cols:
        vals = evidence[col].to_numpy(float)
        rows.append(
            {
                "metric": col,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
        )
    summary = pd.DataFrame(rows)
    concept_summary = (
        concepts.groupby("concept", sort=False)
        .agg(
            active_rate_mean=("top10_severe_active_rate", "mean"),
            active_rate_std=("top10_severe_active_rate", "std"),
            membership_mean=("top10_severe_mean_membership", "mean"),
            membership_std=("top10_severe_mean_membership", "std"),
        )
        .reset_index()
    )
    wide = concepts.pivot(index="split", columns="concept", values="top10_severe_active_rate").reset_index()
    concept_cols = [c for c in wide.columns if c != "split"]
    adjacent_rows = []
    for i in range(1, len(wide)):
        prev = set(wide.loc[i - 1, concept_cols][wide.loc[i - 1, concept_cols] >= 0.5].index)
        cur = set(wide.loc[i, concept_cols][wide.loc[i, concept_cols] >= 0.5].index)
        adjacent_rows.append({"split_pair": f"{wide.loc[i - 1, 'split']}|{wide.loc[i, 'split']}", "concept_jaccard": jaccard(prev, cur)})
    stability = pd.DataFrame(adjacent_rows)
    return summary, concept_summary, stability


def write_report(out: Path, evidence: pd.DataFrame, summary: pd.DataFrame, concept_summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    lines = [
        "# SDS-KB evidence quality",
        "",
        "The evidence-quality diagnostics evaluate the audit channel of SDS-KB separately from the scoring channel.",
        "Positive distance margin means a path is closer to the severe prototype store than to the normal prototype store.",
        "",
        "## Evidence summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Concept frequency in top-risk severe paths",
        "",
        concept_summary.to_markdown(index=False),
        "",
        "## Adjacent-split concept stability",
        "",
        stability.to_markdown(index=False) if not stability.empty else "Smoke run has one split, so adjacent-split stability is not computed.",
        "",
        "## Split-level evidence diagnostics",
        "",
        evidence.to_markdown(index=False),
        "",
    ]
    (out / "sds_kb_evidence_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="2025_07_to_2025_08")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    evidence_rows = []
    concept_frames = []
    case_frames = []
    reliability_frames = []
    for split in splits:
        evidence, concepts, cases, reliability = evaluate_split(split, args.seed, args.max_train_rows)
        evidence_rows.append(evidence)
        concept_frames.append(concepts)
        case_frames.append(cases)
        reliability_frames.append(reliability)
        print(f"{split}: evidence hit={evidence['evidence_hit_rate']:.4f}, margin gap={evidence['separation_median_gap']:.4f}")
    evidence_df = pd.DataFrame(evidence_rows)
    concepts_df = pd.concat(concept_frames, ignore_index=True)
    cases_df = pd.concat(case_frames, ignore_index=True)
    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    summary, concept_summary, stability = summarize(evidence_df, concepts_df)
    evidence_df.to_csv(out / "sds_kb_evidence_quality_split.csv", index=False)
    concepts_df.to_csv(out / "sds_kb_concept_frequency.csv", index=False)
    concept_summary.to_csv(out / "sds_kb_concept_frequency_summary.csv", index=False)
    stability.to_csv(out / "sds_kb_concept_stability.csv", index=False)
    summary.to_csv(out / "sds_kb_evidence_quality_summary.csv", index=False)
    cases_df.to_csv(out / "sds_kb_evidence_cases.csv", index=False)
    reliability_df.to_csv(out / "sds_kb_reliability_bins.csv", index=False)
    write_report(out, evidence_df, summary, concept_summary, stability)


if __name__ == "__main__":
    main()

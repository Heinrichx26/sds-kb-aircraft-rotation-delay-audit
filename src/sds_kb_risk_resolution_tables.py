from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "kbs_risk_resolution"


def mean_summary(df: pd.DataFrame, name_col: str, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, g in df.groupby(name_col, sort=False):
        row = {name_col: name, "splits": g["split"].nunique(), "rows": int(g["rows"].sum()), "positives": int(g["positives"].sum())}
        for metric in metric_cols:
            vals = g[metric].to_numpy(float)
            row[metric] = float(np.nanmean(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def load_score_rows() -> pd.DataFrame:
    tree = pd.read_csv(RESULTS / "full_fepl_validation" / "fepl_rolling_metrics.csv")
    trans = pd.read_csv(RESULTS / "transparent_baseline_validation" / "transparent_baseline_metrics.csv")
    sds = pd.read_csv(RESULTS / "kbs_sds_kb_strengthening" / "sds_kb_ablation_detail.csv")
    kbs_routes = pd.read_csv(RESULTS / "kbs_r5_full_validation" / "kbs_frontier_smoke_detail.csv")
    fepl = pd.read_csv(RESULTS / "stage_constrained_fepl_upgrade" / "stage_constrained_rolling_metrics.csv")

    score_frames = []
    keep_tree = ["LightGBM path baseline", "XGBoost path baseline", "CatBoost path baseline"]
    tree_part = tree[tree["model"].isin(keep_tree)].copy()
    tree_part["model"] = tree_part["model"].str.replace(" path baseline", "", regex=False)
    score_frames.append(tree_part[["split", "model", "rows", "positives", "top10_capture", "auc", "pr_auc", "brier"]])

    sds_part = sds[sds["variant"].eq("Full SDS-KB")].copy()
    sds_part["model"] = "SDS-KB"
    score_frames.append(sds_part[["split", "model", "rows", "positives", "top10_capture", "auc", "pr_auc", "brier"]])

    transparent_rename = {
        "Explainable Boosting Machine": "EBM",
        "Wang-Mendel fuzzy rule classifier": "Wang-Mendel",
    }
    trans_part = trans[trans["model"].isin(transparent_rename)].copy()
    trans_part["model"] = trans_part["model"].map(transparent_rename)
    score_frames.append(trans_part[["split", "model", "rows", "positives", "top10_capture", "auc", "pr_auc", "brier"]])

    fepl_part = fepl[fepl["model"].eq("Stage-constrained FEPL")].copy()
    fepl_part["model"] = "FEPL"
    score_frames.append(fepl_part[["split", "model", "rows", "positives", "top10_capture", "auc", "pr_auc", "brier"]])

    route_rename = {
        "Recovery shortfall reference": "Recovery shortfall",
    }
    route_part = kbs_routes[kbs_routes["route"].isin(route_rename)].copy()
    route_part["model"] = route_part["route"].map(route_rename)
    score_frames.append(route_part[["split", "model", "rows", "positives", "top10_capture", "auc", "pr_auc", "brier"]])
    return pd.concat(score_frames, ignore_index=True)


def paired_tests(score_rows: pd.DataFrame) -> pd.DataFrame:
    base = score_rows[score_rows["model"].eq("SDS-KB")].set_index("split")
    rows: list[dict] = []
    for model in ["LightGBM", "XGBoost", "CatBoost", "EBM", "Wang-Mendel", "FEPL", "Recovery shortfall"]:
        comp = score_rows[score_rows["model"].eq(model)].set_index("split")
        common = sorted(set(base.index).intersection(comp.index))
        for metric, better in [("top10_capture", "higher"), ("pr_auc", "higher"), ("brier", "lower")]:
            delta = base.loc[common, metric].to_numpy(float) - comp.loc[common, metric].to_numpy(float)
            if better == "lower":
                delta = -delta
            nonzero = delta[np.abs(delta) > 1e-12]
            p_value = float(wilcoxon(nonzero, alternative="greater").pvalue) if len(nonzero) else 1.0
            rows.append(
                {
                    "comparison": f"SDS-KB vs {model}",
                    "metric": metric,
                    "better": better,
                    "mean_advantage": float(np.mean(delta)),
                    "wins": int(np.sum(delta > 0)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "splits": len(common),
                    "wilcoxon_p_greater": p_value,
                }
            )
    return pd.DataFrame(rows)


def audit_utility() -> pd.DataFrame:
    evidence = pd.read_csv(RESULTS / "sds_kb_evidence_quality" / "sds_kb_evidence_quality_split.csv")
    full = pd.read_csv(RESULTS / "kbs_sds_kb_strengthening" / "sds_kb_ablation_detail.csv")
    full = full[full["variant"].eq("Full SDS-KB")].copy()
    merged = full.merge(evidence[["split", "evidence_hit_rate"]], on="split", how="left")
    reviewed = np.ceil(0.10 * merged["rows"].to_numpy(float))
    captured = merged["top10_capture"].to_numpy(float) * merged["positives"].to_numpy(float)
    supported = captured * merged["evidence_hit_rate"].to_numpy(float)
    return pd.DataFrame(
        [
            {
                "model": "SDS-KB",
                "reviewed_paths_top10": int(reviewed.sum()),
                "captured_severe_cases": int(round(float(captured.sum()))),
                "evidence_supported_severe_cases": int(round(float(supported.sum()))),
                "severe_cases_per_100_reviews": float(100.0 * captured.sum() / reviewed.sum()),
                "evidence_supported_cases_per_100_reviews": float(100.0 * supported.sum() / reviewed.sum()),
                "mean_evidence_hit_rate": float(np.nanmean(merged["evidence_hit_rate"].to_numpy(float))),
            }
        ]
    )


def role_diagnostics() -> pd.DataFrame:
    detail = pd.read_csv(RESULTS / "kbs_sds_kb_strengthening" / "sds_kb_ablation_detail.csv")
    keep = ["Full SDS-KB", "No simulated severe store", "No prototype affinity", "No fuzzy concepts", "Prototype-only scorer", "No monotone calibration"]
    part = detail[detail["variant"].isin(keep)].copy()
    return mean_summary(part, "variant", ["top10_capture", "auc", "pr_auc", "brier"])


def cross_year() -> pd.DataFrame:
    path = RESULTS / "kbs_sds_kb_cross_year" / "sds_kb_cross_year_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_report(score: pd.DataFrame, tests: pd.DataFrame, audit: pd.DataFrame, roles: pd.DataFrame, cross: pd.DataFrame) -> None:
    lines = [
        "# SDS-KB risk-resolution tables",
        "",
        "These tables support the revised positioning: score ranking is reported separately from knowledge-base audit evidence.",
        "",
        "## Rolling score references",
        "",
        score.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired split-level tests",
        "",
        tests.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Audit utility",
        "",
        audit.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Component role diagnostics",
        "",
        roles.to_markdown(index=False, floatfmt=".4f"),
    ]
    if not cross.empty:
        lines.extend(["", "## Cross-year transfer", "", cross.to_markdown(index=False, floatfmt=".4f")])
    (OUT / "risk_resolution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score_rows = load_score_rows()
    score = mean_summary(score_rows, "model", ["top10_capture", "auc", "pr_auc", "brier"])
    tests = paired_tests(score_rows)
    audit = audit_utility()
    roles = role_diagnostics()
    cross = cross_year()
    score.to_csv(OUT / "rolling_score_references.csv", index=False)
    tests.to_csv(OUT / "paired_split_tests.csv", index=False)
    audit.to_csv(OUT / "audit_utility.csv", index=False)
    roles.to_csv(OUT / "component_role_diagnostics.csv", index=False)
    if not cross.empty:
        cross.to_csv(OUT / "cross_year_transfer.csv", index=False)
    write_report(score, tests, audit, roles, cross)
    print((OUT / "risk_resolution_report.md").resolve())


if __name__ == "__main__":
    main()

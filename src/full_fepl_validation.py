from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smoke_fepl_topic import (
    build_rotation_paths,
    fit_fepl,
    fit_tree_models,
    metric_dict,
    rule_audit,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "full_fepl_validation"

BASELINE_MODELS = [
    "LightGBM path baseline",
    "XGBoost path baseline",
    "CatBoost path baseline",
]
SIMPLE_MODELS = ["Previous arrival delay", "Recovery shortfall", "Raw FEPL max membership"]


def pair_tag(year: int, train_month: int, test_month: int) -> str:
    return f"{year}_{train_month:02d}_to_{year}_{test_month:02d}"


def run_one_split(
    year: int,
    train_month: int,
    test_month: int,
    carriers: set[str],
    out: Path,
    max_train_rows: int,
    seed: int,
    reuse_panel: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict]:
    tag = pair_tag(year, train_month, test_month)
    split_out = out / "splits" / tag
    split_out.mkdir(parents=True, exist_ok=True)
    panel_path = split_out / "fepl_rotation_path_panel.csv"
    if reuse_panel and panel_path.exists():
        paths = pd.read_csv(panel_path, parse_dates=["dep_dt", "arr_dt", "prev_arr_dt"])
    else:
        paths = build_rotation_paths([(year, train_month), (year, test_month)], carriers, split_out)

    train_ym = f"{year}_{train_month:02d}"
    test_ym = f"{year}_{test_month:02d}"
    train = paths[paths["ym"].eq(train_ym)].copy()
    test = paths[paths["ym"].eq(test_ym)].copy()
    y_test = test["severe_late_aircraft"].to_numpy(int)

    fepl_score, fepl_prob, weights = fit_fepl(train, test)
    rows = [
        {"split": tag, "train_month": train_ym, "test_month": test_ym, "model": "FEPL fuzzy path lattice", **metric_dict(y_test, fepl_score, fepl_prob)},
        {"split": tag, "train_month": train_ym, "test_month": test_ym, "model": "Previous arrival delay", **metric_dict(y_test, test["prev_arr_delay"].to_numpy(float), None)},
        {"split": tag, "train_month": train_ym, "test_month": test_ym, "model": "Recovery shortfall", **metric_dict(y_test, test["recovery_shortfall"].to_numpy(float), None)},
        {"split": tag, "train_month": train_ym, "test_month": test_ym, "model": "Raw FEPL max membership", **metric_dict(y_test, test["mu_fepl_path"].to_numpy(float), test["mu_fepl_path"].to_numpy(float))},
    ]
    tree_preds = fit_tree_models(train, test, seed, max_train_rows)
    for model, pred in tree_preds.items():
        rows.append({"split": tag, "train_month": train_ym, "test_month": test_ym, "model": model, **metric_dict(y_test, pred, pred)})

    metrics = pd.DataFrame(rows).sort_values(["top10_capture", "auc"], ascending=False)
    metrics.to_csv(split_out / "split_metrics.csv", index=False)
    rules = rule_audit(test, fepl_score, split_out)
    rules.insert(0, "split", tag)
    metadata = {
        "split": tag,
        "train_month": train_ym,
        "test_month": test_ym,
        "carriers": sorted(carriers),
        "train_paths": int(len(train)),
        "test_paths": int(len(test)),
        "test_positives": int(y_test.sum()),
        "positive_definition": "LateAircraftDelay >= 30 minutes",
    }
    (split_out / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, rules, weights, metadata


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            splits=("split", "nunique"),
            total_rows=("rows", "sum"),
            total_positives=("positives", "sum"),
            mean_top10_capture=("top10_capture", "mean"),
            mean_top10_lift=("top10_lift", "mean"),
            mean_auc=("auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values(["mean_top10_capture", "mean_auc"], ascending=False)
    )


def bootstrap_split_diffs(metrics: pd.DataFrame, out: Path, n_bootstrap: int, seed: int) -> pd.DataFrame:
    wide = metrics.pivot_table(index="split", columns="model", values=["top10_capture", "auc", "pr_auc", "brier"], aggfunc="first")
    splits = wide.index.to_numpy()
    rng = np.random.default_rng(seed)
    comparisons = [
        ("fepl_minus_lightgbm", "FEPL fuzzy path lattice", "LightGBM path baseline"),
        ("fepl_minus_catboost", "FEPL fuzzy path lattice", "CatBoost path baseline"),
        ("fepl_minus_xgboost", "FEPL fuzzy path lattice", "XGBoost path baseline"),
    ]
    rows = []
    for b in range(n_bootstrap):
        sample_splits = rng.choice(splits, size=len(splits), replace=True)
        sample = wide.loc[sample_splits]
        for metric in ["top10_capture", "auc", "pr_auc", "brier"]:
            for name, left, right in comparisons:
                rows.append(
                    {
                        "bootstrap": b,
                        "comparison": name,
                        "metric": metric,
                        "diff": float((sample[(metric, left)] - sample[(metric, right)]).mean()),
                    }
                )
    samples = pd.DataFrame(rows)
    samples.to_csv(out / "fepl_bootstrap_split_samples.csv", index=False)

    summary_rows = []
    for metric in ["top10_capture", "auc", "pr_auc", "brier"]:
        for name, left, right in comparisons:
            actual = float((wide[(metric, left)] - wide[(metric, right)]).mean())
            part = samples[samples["comparison"].eq(name) & samples["metric"].eq(metric)]["diff"]
            summary_rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "actual_diff": actual,
                    "ci_low": float(part.quantile(0.025)),
                    "ci_high": float(part.quantile(0.975)),
                    "prob_diff_gt_0": float((part > 0).mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "fepl_bootstrap_split_differences.csv", index=False)
    return summary


def rule_stability(rules: pd.DataFrame, out: Path) -> pd.DataFrame:
    top_rules = {}
    for split, part in rules.groupby("split"):
        top_rules[split] = set(part.sort_values("share_of_all_positive_cases", ascending=False).head(3)["rule"].tolist())
    rows = []
    splits = list(top_rules)
    for i in range(1, len(splits)):
        prev = top_rules[splits[i - 1]]
        cur = top_rules[splits[i]]
        rows.append(
            {
                "previous_split": splits[i - 1],
                "current_split": splits[i],
                "top3_jaccard": len(prev & cur) / max(len(prev | cur), 1),
                "previous_rules": "; ".join(sorted(prev)),
                "current_rules": "; ".join(sorted(cur)),
            }
        )
    stability = pd.DataFrame(
        rows,
        columns=["previous_split", "current_split", "top3_jaccard", "previous_rules", "current_rules"],
    )
    stability.to_csv(out / "fepl_rule_stability.csv", index=False)
    return stability


def weight_stability(weight_rows: list[dict], out: Path) -> pd.DataFrame:
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(out / "fepl_split_weights.csv", index=False)
    summary = (
        weights.drop(columns=["split"])
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "membership"})
    )
    summary.to_csv(out / "fepl_weight_stability.csv", index=False)
    return summary


def run_carrier_holdout(
    year: int,
    train_months: list[int],
    carriers: set[str],
    out: Path,
    reuse_panel: bool,
) -> pd.DataFrame:
    rows = []
    for train_month in train_months:
        test_month = train_month + 1
        tag = pair_tag(year, train_month, test_month)
        panel_path = out / "splits" / tag / "fepl_rotation_path_panel.csv"
        if reuse_panel and panel_path.exists():
            paths = pd.read_csv(panel_path)
        else:
            paths = build_rotation_paths([(year, train_month), (year, test_month)], carriers, out / "splits" / tag)
        train_ym = f"{year}_{train_month:02d}"
        test_ym = f"{year}_{test_month:02d}"
        for carrier in sorted(carriers):
            train = paths[paths["ym"].eq(train_ym) & paths["carrier"].ne(carrier)].copy()
            test = paths[paths["ym"].eq(test_ym) & paths["carrier"].eq(carrier)].copy()
            if train.empty or test.empty or test["severe_late_aircraft"].sum() <= 0:
                continue
            fepl_score, fepl_prob, _ = fit_fepl(train, test)
            y_test = test["severe_late_aircraft"].to_numpy(int)
            rows.append(
                {
                    "split": tag,
                    "heldout_carrier": carrier,
                    "model": "FEPL fuzzy path lattice",
                    **metric_dict(y_test, fepl_score, fepl_prob),
                }
            )
            rows.append(
                {
                    "split": tag,
                    "heldout_carrier": carrier,
                    "model": "Recovery shortfall",
                    **metric_dict(y_test, test["recovery_shortfall"].to_numpy(float), None),
                }
            )
    holdout = pd.DataFrame(rows)
    holdout.to_csv(out / "fepl_carrier_holdout_metrics.csv", index=False)
    summary = (
        holdout.groupby("model", as_index=False)
        .agg(
            folds=("split", "count"),
            mean_top10_capture=("top10_capture", "mean"),
            mean_auc=("auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values("mean_top10_capture", ascending=False)
    )
    summary.to_csv(out / "fepl_carrier_holdout_summary.csv", index=False)
    return summary


def markdown_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(formats.get(col, "{}").format(row[col]) for col in columns) + " |")
    return lines


def write_report(out: Path, metadata: dict) -> None:
    summary = pd.read_csv(out / "fepl_rolling_summary.csv")
    diffs = pd.read_csv(out / "fepl_bootstrap_split_differences.csv")
    rules = pd.read_csv(out / "fepl_rule_stability.csv")
    weights = pd.read_csv(out / "fepl_weight_stability.csv")
    holdout_paths = [
        ("Carrier Holdout", out / "fepl_carrier_holdout_summary.csv"),
        ("Fixed-Rule Carrier Holdout", out / "fepl_carrier_holdout_fixed_rule_summary.csv"),
    ]
    lines = [
        "# FEPL Full Validation",
        "",
        f"Year: {metadata['year']}; train-test month pairs: {metadata['pairs']}.",
        f"Carriers: {', '.join(metadata['carriers'])}.",
        f"Total test paths: {metadata['total_test_paths']:,}; total severe late-aircraft cases: {metadata['total_test_positives']:,}.",
        "",
        "## Rolling Month Metrics",
        "",
    ]
    cols = ["model", "splits", "mean_top10_capture", "mean_auc", "mean_pr_auc", "mean_brier"]
    lines.extend(
        markdown_table(
            summary[cols],
            cols,
            {
                "model": "{}",
                "splits": "{:.0f}",
                "mean_top10_capture": "{:.4f}",
                "mean_auc": "{:.4f}",
                "mean_pr_auc": "{:.4f}",
                "mean_brier": "{:.4f}",
            },
        )
    )
    lines += ["", "## Bootstrap Split Differences", ""]
    cols_diff = ["comparison", "metric", "actual_diff", "ci_low", "ci_high", "prob_diff_gt_0"]
    lines.extend(
        markdown_table(
            diffs[cols_diff],
            cols_diff,
            {
                "comparison": "{}",
                "metric": "{}",
                "actual_diff": "{:.5f}",
                "ci_low": "{:.5f}",
                "ci_high": "{:.5f}",
                "prob_diff_gt_0": "{:.3f}",
            },
        )
    )
    lines += ["", "## Rule Stability", ""]
    if not rules.empty:
        lines.append(f"Mean adjacent top-3 rule Jaccard similarity: {rules['top3_jaccard'].mean():.3f}.")
    lines += ["", "## Weight Stability", ""]
    cols_weights = ["membership", "mean", "std", "min", "max"]
    lines.extend(
        markdown_table(
            weights[cols_weights],
            cols_weights,
            {"membership": "{}", "mean": "{:.3f}", "std": "{:.3f}", "min": "{:.3f}", "max": "{:.3f}"},
        )
    )
    for holdout_title, holdout_path in holdout_paths:
        if not holdout_path.exists():
            continue
        holdout = pd.read_csv(holdout_path)
        lines += ["", f"## {holdout_title}", ""]
        cols_holdout = ["model", "folds", "mean_top10_capture", "mean_auc", "mean_pr_auc", "mean_brier"]
        lines.extend(
            markdown_table(
                holdout[cols_holdout],
                cols_holdout,
                {
                    "model": "{}",
                    "folds": "{:.0f}",
                    "mean_top10_capture": "{:.4f}",
                    "mean_auc": "{:.4f}",
                    "mean_pr_auc": "{:.4f}",
                    "mean_brier": "{:.4f}",
                },
            )
        )
    (out / "fepl_full_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--carriers", default="WN,DL,AA,OO,UA")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-panel", action="store_true")
    parser.add_argument("--skip-carrier-holdout", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        metadata = json.loads((out / "fepl_full_metadata.json").read_text(encoding="utf-8"))
        write_report(out, metadata)
        return

    carriers = {item.strip().upper() for item in args.carriers.split(",") if item.strip()}
    metrics_list = []
    rules_list = []
    weight_rows = []
    split_meta = []
    for month in range(args.start_month, args.end_month + 1):
        metrics, rules, weights, metadata = run_one_split(
            args.year,
            month,
            month + 1,
            carriers,
            out,
            args.max_train_rows,
            args.seed,
            args.reuse_panel,
        )
        metrics_list.append(metrics)
        rules_list.append(rules)
        weight_rows.append({"split": metadata["split"], **weights})
        split_meta.append(metadata)

    metrics = pd.concat(metrics_list, ignore_index=True)
    rules = pd.concat(rules_list, ignore_index=True)
    metrics.to_csv(out / "fepl_rolling_metrics.csv", index=False)
    rules.to_csv(out / "fepl_rolling_rule_audit.csv", index=False)
    summarize_metrics(metrics).to_csv(out / "fepl_rolling_summary.csv", index=False)
    bootstrap_split_diffs(metrics, out, args.bootstrap, args.seed)
    rule_stability(rules, out)
    weight_stability(weight_rows, out)

    if not args.skip_carrier_holdout:
        holdout_months = list(range(args.start_month, args.end_month + 1))
        run_carrier_holdout(args.year, holdout_months, carriers, out, reuse_panel=True)

    metadata = {
        "year": args.year,
        "pairs": [item["split"] for item in split_meta],
        "carriers": sorted(carriers),
        "total_test_paths": int(sum(item["test_paths"] for item in split_meta)),
        "total_test_positives": int(sum(item["test_positives"] for item in split_meta)),
        "max_train_rows_for_tree_baselines": args.max_train_rows,
    }
    (out / "fepl_full_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out, metadata)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ARTICLE_FIG_DIR = PROJECT / "article" / "manuscript" / "figures"
FIG_DIR = ARTICLE_FIG_DIR if ARTICLE_FIG_DIR.exists() else PROJECT / "figures"
FULL = PROJECT / "results" / "kbs_r5_full_validation"
AIRPORT = PROJECT / "results" / "kbs_r5_airport_holdout"
STRENGTH = PROJECT / "results" / "kbs_sds_kb_strengthening"
TRANSFER = PROJECT / "results" / "transfer_score_audit_baselines"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.35, alpha=0.30)


def label_bars(ax, bars, fmt="{:.3f}") -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.018 * span, ymax - 0.035 * span),
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=6.0,
            clip_on=True,
        )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    if (STRENGTH / "sds_kb_ablation_detail.csv").exists():
        detail = pd.read_csv(STRENGTH / "sds_kb_ablation_detail.csv")
        r5 = detail[detail["variant"].eq("Full SDS-KB")].copy()
        sds_pr_auc = float(
            pd.read_csv(STRENGTH / "sds_kb_ablation_summary.csv")
            .loc[lambda x: x["variant"].eq("Full SDS-KB"), "pr_auc_mean"]
            .iloc[0]
        )
        recovery_pr_auc = float(
            pd.read_csv(FULL / "kbs_frontier_smoke_summary.csv")
            .loc[lambda x: x["route"].eq("Recovery shortfall reference"), "mean_pr_auc"]
            .iloc[0]
        )
    else:
        detail = pd.read_csv(FULL / "kbs_frontier_smoke_detail.csv")
        summary = pd.read_csv(FULL / "kbs_frontier_smoke_summary.csv")
        r5 = detail[detail["route"].eq("R5 severe-delay simulation knowledge base")].copy()
        sds_pr_auc = float(summary.loc[summary["route"].eq("R5 severe-delay simulation knowledge base"), "mean_pr_auc"].iloc[0])
        recovery_pr_auc = float(summary.loc[summary["route"].eq("Recovery shortfall reference"), "mean_pr_auc"].iloc[0])
    transfer = pd.read_csv(TRANSFER / "transfer_score_audit_baselines_summary.csv")

    r5["test_month"] = r5["split"].str[-7:].str.replace("2025_", "")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.45), constrained_layout=False)

    ax = axes[0, 0]
    x = range(len(r5))
    ax.plot(x, r5["top10_capture"], marker="o", linewidth=1.25, markersize=3, color="#0F766E", label="T10 capture")
    ax.plot(x, r5["pr_auc"], marker="s", linewidth=1.25, markersize=3, color="#2563EB", label="PR-AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[-2:] for m in r5["test_month"]])
    ax.set_ylim(0.84, 1.01)
    ax.set_title("(a) Rolling PE-KB validation")
    ax.set_xlabel("Held-out month")
    ax.set_ylabel("Metric")
    finish(ax)

    ax = axes[0, 1]
    rows = [
        ("LightGBM", 0.9359),
        ("PE-KB", sds_pr_auc),
        ("EBM", 0.8629),
        ("Wang-Mendel", 0.8276),
        ("FEPL", 0.8092),
        ("Recovery", recovery_pr_auc),
    ]
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=["#94A3B8", "#0F766E", "#5B8DB8", "#A8A29E", "#B6A6CA", "#CBD5E1"])
    ax.set_ylim(0.70, 0.96)
    ax.set_title("(b) PR-AUC score references")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=22)
    label_bars(ax, bars)
    finish(ax)

    ax = axes[1, 0]
    protocols = [("airport", "Airport"), ("carrier", "Carrier"), ("cross_year", "Cross-year")]
    pe_vals = []
    lgbm_vals = []
    for key, _ in protocols:
        pe_vals.append(
            float(
                transfer.loc[
                    transfer["protocol"].eq(key) & transfer["queue"].eq("PE-KB queue"),
                    "pooled_evidence_supported_cases_per_100_reviews",
                ].iloc[0]
            )
        )
        lgbm_vals.append(
            float(
                transfer.loc[
                    transfer["protocol"].eq(key) & transfer["queue"].eq("LightGBM score + PE-KB evidence"),
                    "pooled_evidence_supported_cases_per_100_reviews",
                ].iloc[0]
            )
        )
    x2 = list(range(len(protocols)))
    width = 0.34
    bars1 = ax.bar([x - width / 2 for x in x2], pe_vals, width=width, color="#0F766E", label="PE-KB")
    bars2 = ax.bar([x + width / 2 for x in x2], lgbm_vals, width=width, color="#94A3B8", label="LightGBM+PE-KB")
    ax.set_xticks(x2)
    ax.set_xticklabels([label for _, label in protocols])
    ax.set_ylim(43.6, 47.2)
    ax.set_title("(c) Transfer supported yield")
    ax.set_ylabel("Supported severe per 100 reviews")
    label_bars(ax, bars1, fmt="{:.2f}")
    label_bars(ax, bars2, fmt="{:.2f}")
    ax.legend(frameon=False, loc="upper left", fontsize=6.0)
    finish(ax)

    ax = axes[1, 1]
    audit_path = PROJECT / "results" / "kbs_risk_resolution" / "audit_utility.csv"
    workload_path = PROJECT / "results" / "audit_workload_summary" / "audit_workload_summary.csv"
    if audit_path.exists() and workload_path.exists():
        audit = pd.read_csv(audit_path).iloc[0]
        workload_summary = pd.read_csv(workload_path)

        def workload_value(model_name: str) -> float:
            return float(
                workload_summary.loc[
                    workload_summary["model"].eq(model_name),
                    "severe_cases_per_100_reviews",
                ].iloc[0]
            )

        workload = [
            ("PE-KB", float(audit["severe_cases_per_100_reviews"])),
            ("FEPL", workload_value("FEPL fuzzy path rule model")),
            ("Wang-Mendel", workload_value("Wang-Mendel fuzzy rule classifier")),
            ("Recovery", workload_value("Recovery shortfall")),
        ]
    else:
        workload = [
            ("PE-KB", 48.78),
            ("FEPL", 46.86),
            ("Wang-Mendel", 46.96),
            ("Recovery", 46.48),
        ]
    bars = ax.bar([r[0] for r in workload], [r[1] for r in workload], color=["#0F766E", "#B6A6CA", "#A8A29E", "#CBD5E1"])
    ax.set_ylim(45.8, 49.4)
    ax.set_title("(d) Top-10% review yield")
    ax.set_ylabel("Severe cases per 100 reviews")
    label_bars(ax, bars, fmt="{:.2f}")
    finish(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.16, hspace=0.50, wspace=0.27)
    for stem in ["sds_kb_validation_summary", "fepl_validation_summary"]:
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(FIG_DIR / f"{stem}.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()

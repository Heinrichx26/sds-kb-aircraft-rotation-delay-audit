from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "article" / "manuscript" / "figures"
EVIDENCE = PROJECT / "results" / "sds_kb_evidence_quality"
CALIBRATION = PROJECT / "results" / "calibration_fairness_full40k"


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
    ax.grid(True, linewidth=0.35, alpha=0.32)


def risk_tier_rows() -> pd.DataFrame:
    rel = pd.read_csv(EVIDENCE / "sds_kb_reliability_bins.csv")
    groups = [
        (-1, 0, "T1\nlowest"),
        (0, 2, "T2"),
        (2, 7, "T3"),
        (7, 11, "T4"),
        (11, 14, "T5\nhighest"),
    ]
    total_rows = rel["rows"].sum()
    total_severe = (rel["observed_rate"] * rel["rows"]).sum()
    rows = []
    for low, high, label in groups:
        part = rel[(rel["bin"] > low) & (rel["bin"] <= high)].copy()
        n = part["rows"].sum()
        severe = (part["observed_rate"] * part["rows"]).sum()
        rows.append(
            {
                "tier": label,
                "rows": n,
                "row_share": n / total_rows,
                "severe_rate": severe / n,
                "severe_share": severe / total_severe,
            }
        )
    return pd.DataFrame(rows)


def plot_risk_tiers() -> None:
    tiers = risk_tier_rows()
    colors = ["#E2E8F0", "#B6CBD0", "#7DA9A6", "#3D817A", "#0F766E"]

    fig, ax = plt.subplots(figsize=(5.65, 2.05), constrained_layout=False)
    xlabels = [f"{row.tier}\nshare {row.row_share * 100:.1f}%" for row in tiers.itertuples(index=False)]
    bars = ax.bar(xlabels, tiers["severe_rate"], color=colors, edgecolor="#334155", linewidth=0.4)
    overall = (tiers["severe_rate"] * tiers["rows"]).sum() / tiers["rows"].sum()
    ax.axhline(overall, color="#64748B", linewidth=0.8, linestyle="--", label="Overall rate")
    ax.set_ylim(0, 0.98)
    ax.set_ylabel("Observed severe-case rate")
    ax.set_xlabel("PE-KB monotone risk tier")
    finish(ax)
    for bar, row in zip(bars, tiers.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(max(row.severe_rate + 0.035, 0.075), 0.94),
            f"{row.severe_rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=True,
        )
    ax.legend(frameon=False, loc="upper left", borderaxespad=0.1)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.26)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "sds_kb_risk_tiers.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(FIG_DIR / "sds_kb_risk_tiers.png", dpi=260, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.2f}") -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02 * span, ymax - 0.035 * span),
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=6.2,
            clip_on=True,
        )


def plot_evidence_quality() -> None:
    split = pd.read_csv(EVIDENCE / "sds_kb_evidence_quality_split.csv")
    concept = pd.read_csv(EVIDENCE / "sds_kb_concept_frequency_summary.csv")
    stability = pd.read_csv(EVIDENCE / "sds_kb_concept_stability.csv")
    months = [s[-2:] for s in split["split"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55), constrained_layout=False)

    ax = axes[0, 0]
    ax.plot(months, split["severe_margin_median"], marker="o", linewidth=1.2, markersize=3, color="#0F766E", label="Severe")
    ax.plot(months, split["normal_margin_median"], marker="s", linewidth=1.2, markersize=3, color="#94A3B8", label="Normal")
    ax.set_title("(a) Prototype margin by label")
    ax.set_xlabel("Held-out month")
    ax.set_ylabel("Median $d^0-d^+$")
    finish(ax)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.18), borderaxespad=0.1)

    ax = axes[0, 1]
    ax.plot(months, split["evidence_hit_rate"], marker="o", linewidth=1.2, markersize=3, color="#2563EB")
    ax.set_ylim(0.94, 1.00)
    ax.set_title("(b) Top-risk evidence hit rate")
    ax.set_xlabel("Held-out month")
    ax.set_ylabel("Rate")
    finish(ax)

    ax = axes[1, 0]
    concept = concept.sort_values("active_rate_mean", ascending=False)
    bars = ax.bar(concept["concept"], concept["active_rate_mean"], yerr=concept["active_rate_std"], color="#0F766E", alpha=0.88, capsize=2)
    ax.set_ylim(0.0, 1.04)
    ax.set_title("(c) Active concepts in top-risk severe paths")
    ax.set_ylabel("Active rate")
    ax.tick_params(axis="x", rotation=25)
    label_bars(ax, bars, "{:.2f}")
    finish(ax)

    ax = axes[1, 1]
    bars = ax.bar(
        ["Nearest severe", "Nearest normal"],
        [
            split["nearest_severe_delay_median_top10_severe"].mean(),
            split["nearest_normal_delay_median_top10_severe"].mean(),
        ],
        color=["#0F766E", "#94A3B8"],
    )
    ax.set_ylim(0, 64)
    ax.set_title("(d) Nearest-prototype delay consistency")
    ax.set_ylabel("Median late-aircraft delay (min)")
    if not stability.empty:
        ax.text(
            0.78,
            0.86,
            f"Concept Jaccard = {stability['concept_jaccard'].mean():.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.6},
        )
    label_bars(ax, bars, "{:.1f}")
    finish(ax)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.16, hspace=0.58, wspace=0.29)
    fig.savefig(FIG_DIR / "sds_kb_evidence_quality.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG_DIR / "sds_kb_evidence_quality.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    set_style()
    plot_risk_tiers()
    plot_evidence_quality()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kbs_frontier_smoke_tests import calibrate, fit_lgbm, read_split, split_fit_calibration, top_capture


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "kbs_sds_kb_strengthening"
YEAR_PANEL_DIR = PROJECT / "data" / "interim" / "fepl_year_panels"

RAW_COLS = [
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
]

FUZZY_COLS = [
    "mu_upstream_delay",
    "mu_prev_late_aircraft",
    "mu_tight_turnaround",
    "mu_recovery_shortfall",
    "mu_origin_chain_density",
    "mu_buffer_failure_path",
    "mu_late_recurrence_path",
    "mu_congested_carryover_path",
]

CASE_COLS = [
    "carrier",
    "prev_origin",
    "origin",
    "dest",
    "prev_arr_delay",
    "prev_late_aircraft_delay",
    "scheduled_turnaround",
    "actual_turnaround",
    "recovery_shortfall",
    "origin_hour_chain_density",
    "mu_buffer_failure_path",
    "mu_late_recurrence_path",
    "mu_congested_carryover_path",
    "LateAircraftDelay",
    "severe_late_aircraft",
]


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def metric_dict(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prob_arr = np.clip(np.asarray(score if prob is None else prob, dtype=float), 0.0, 1.0)
    t10, hit = top_capture(y, score)
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "top10_capture": t10,
        "top10_hit_rate": hit,
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, prob_arr)),
    }


def add_prototype_features_custom(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    max_proto: int = 4500,
    neighbors: int = 5,
    use_fuzzy: bool = True,
    return_neighbors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, np.ndarray] | None]:
    base_cols = available(train, RAW_COLS)
    fuzzy_cols = available(train, FUZZY_COLS) if use_fuzzy else []
    proto_cols = [*base_cols, *fuzzy_cols]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[proto_cols].fillna(0.0).to_numpy(float))
    test_x = scaler.transform(test[proto_cols].fillna(0.0).to_numpy(float))
    severe_idx = np.where(train["severe_late_aircraft"].to_numpy(int) == 1)[0]
    normal_idx = np.where(train["severe_late_aircraft"].to_numpy(int) == 0)[0]
    rng = np.random.default_rng(seed)
    severe_idx = rng.choice(severe_idx, size=min(len(severe_idx), max_proto), replace=False)
    normal_idx = rng.choice(normal_idx, size=min(len(normal_idx), max_proto), replace=False)
    k_sev = min(neighbors, len(severe_idx))
    k_norm = min(neighbors, len(normal_idx))
    severe_nn = NearestNeighbors(n_neighbors=k_sev, algorithm="auto").fit(train_x[severe_idx])
    normal_nn = NearestNeighbors(n_neighbors=k_norm, algorithm="auto").fit(train_x[normal_idx])
    train_d_sev, train_i_sev = severe_nn.kneighbors(train_x)
    train_d_norm, train_i_norm = normal_nn.kneighbors(train_x)
    test_d_sev, test_i_sev = severe_nn.kneighbors(test_x)
    test_d_norm, test_i_norm = normal_nn.kneighbors(test_x)
    train = train.copy()
    test = test.copy()
    for df, d_sev, d_norm in [(train, train_d_sev, train_d_norm), (test, test_d_sev, test_d_norm)]:
        df["kb_dist_severe"] = d_sev.mean(axis=1)
        df["kb_dist_normal"] = d_norm.mean(axis=1)
        df["kb_distance_margin"] = df["kb_dist_normal"] - df["kb_dist_severe"]
        df["kb_severe_affinity"] = 1.0 / (1.0 + df["kb_dist_severe"])
        df["kb_normal_affinity"] = 1.0 / (1.0 + df["kb_dist_normal"])
    neighbor_info = None
    if return_neighbors:
        neighbor_info = {
            "severe_sample_idx": severe_idx,
            "normal_sample_idx": normal_idx,
            "test_nearest_severe_train_idx": severe_idx[test_i_sev[:, 0]],
            "test_nearest_normal_train_idx": normal_idx[test_i_norm[:, 0]],
            "test_nearest_severe_dist": test_d_sev[:, 0],
            "test_nearest_normal_dist": test_d_norm[:, 0],
            "prototype_columns": np.asarray(proto_cols),
        }
    return train, test, ["kb_dist_severe", "kb_dist_normal", "kb_distance_margin", "kb_severe_affinity", "kb_normal_affinity"], neighbor_info


def augment_training(train: pd.DataFrame, feature_cols: list[str], seed: int, multiplier: float) -> pd.DataFrame:
    severe = train[train["severe_late_aircraft"].eq(1)].copy()
    if severe.empty or multiplier <= 0:
        return train
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(severe) * multiplier)))
    keep = severe.sample(n, replace=True, random_state=seed).copy()
    for col in feature_cols:
        arr = pd.to_numeric(severe[col], errors="coerce").fillna(0.0).to_numpy(float)
        scale = max(float(np.nanstd(arr)), 1.0) * 0.035
        keep[col] = pd.to_numeric(keep[col], errors="coerce").fillna(0.0) + rng.normal(0.0, scale, size=len(keep))
    keep["severe_late_aircraft"] = 1
    return pd.concat([train, keep], ignore_index=True)


def direct_prototype_score(df: pd.DataFrame) -> np.ndarray:
    margin = df["kb_distance_margin"].to_numpy(float)
    affinity_gap = df["kb_severe_affinity"].to_numpy(float) - df["kb_normal_affinity"].to_numpy(float)
    return margin + affinity_gap


def run_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    split: str,
    variant: str,
    seed: int,
    max_rows: int,
    max_proto: int,
    neighbors: int,
    simulation_multiplier: float,
    use_prototypes: bool,
    use_fuzzy: bool,
    use_simulation: bool,
    use_tree: bool,
    use_calibration: bool = True,
    n_estimators: int = 280,
) -> dict[str, float | str]:
    train_work = train.copy()
    test_work = test.copy()
    kb_cols: list[str] = []
    if use_prototypes:
        train_work, test_work, kb_cols, _ = add_prototype_features_custom(
            train_work,
            test_work,
            seed=seed,
            max_proto=max_proto,
            neighbors=neighbors,
            use_fuzzy=use_fuzzy,
        )
    raw_cols = available(train_work, RAW_COLS)
    fuzzy_cols = available(train_work, FUZZY_COLS) if use_fuzzy else []
    feature_cols = [*raw_cols, *fuzzy_cols, *kb_cols]
    y_test = test_work["severe_late_aircraft"].to_numpy(int)
    if not use_tree:
        fit_df, cal_df = split_fit_calibration(train_work, seed, max_rows)
        cal_score = direct_prototype_score(cal_df)
        test_score = direct_prototype_score(test_work)
        test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
    else:
        fit_df, cal_df = split_fit_calibration(train_work, seed, max_rows)
        if use_simulation:
            fit_df = augment_training(fit_df, feature_cols, seed, simulation_multiplier)
        model = fit_lgbm(
            fit_df[feature_cols].fillna(0.0),
            fit_df["severe_late_aircraft"].to_numpy(int),
            seed,
            n_estimators=n_estimators,
        )
        cal_score = model.predict_proba(cal_df[feature_cols].fillna(0.0))[:, 1]
        test_score = model.predict_proba(test_work[feature_cols].fillna(0.0))[:, 1]
        if use_calibration:
            test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
        else:
            test_prob = np.clip(test_score, 0.0, 1.0)
    return {
        "split": split,
        "variant": variant,
        "use_prototypes": use_prototypes,
        "use_fuzzy": use_fuzzy,
        "use_simulation": use_simulation,
        "use_tree": use_tree,
        "use_calibration": use_calibration,
        "neighbors": neighbors if use_prototypes else 0,
        "max_proto": max_proto if use_prototypes else 0,
        "simulation_multiplier": simulation_multiplier if use_simulation else 0.0,
        "feature_count": len(feature_cols),
        **metric_dict(y_test, test_score, test_prob),
    }


def ablation_variants() -> list[dict]:
    return [
        {"variant": "Full SDS-KB", "use_prototypes": True, "use_fuzzy": True, "use_simulation": True, "use_tree": True, "use_calibration": True, "simulation_multiplier": 1.0, "neighbors": 5, "max_proto": 4500},
        {"variant": "No simulated severe store", "use_prototypes": True, "use_fuzzy": True, "use_simulation": False, "use_tree": True, "use_calibration": True, "simulation_multiplier": 0.0, "neighbors": 5, "max_proto": 4500},
        {"variant": "No prototype affinity", "use_prototypes": False, "use_fuzzy": True, "use_simulation": True, "use_tree": True, "use_calibration": True, "simulation_multiplier": 1.0, "neighbors": 0, "max_proto": 0},
        {"variant": "No fuzzy concepts", "use_prototypes": True, "use_fuzzy": False, "use_simulation": True, "use_tree": True, "use_calibration": True, "simulation_multiplier": 1.0, "neighbors": 5, "max_proto": 4500},
        {"variant": "Prototype-only scorer", "use_prototypes": True, "use_fuzzy": True, "use_simulation": False, "use_tree": False, "use_calibration": True, "simulation_multiplier": 0.0, "neighbors": 5, "max_proto": 4500},
        {"variant": "No monotone calibration", "use_prototypes": True, "use_fuzzy": True, "use_simulation": True, "use_tree": True, "use_calibration": False, "simulation_multiplier": 1.0, "neighbors": 5, "max_proto": 4500},
    ]


def sensitivity_variants() -> list[dict]:
    variants: list[dict] = []
    for mult in [0.5, 1.0, 1.5]:
        variants.append({"variant": f"Simulation multiplier {mult:g}", "simulation_multiplier": mult, "neighbors": 5, "max_proto": 4500})
    for k in [1, 3, 10]:
        variants.append({"variant": f"Prototype neighbors {k}", "simulation_multiplier": 1.0, "neighbors": k, "max_proto": 4500})
    for max_proto in [1500, 3000]:
        variants.append({"variant": f"Prototype store {max_proto}", "simulation_multiplier": 1.0, "neighbors": 5, "max_proto": max_proto})
    return variants


def summarize(detail: pd.DataFrame, group_col: str = "variant") -> pd.DataFrame:
    rows = []
    for name, g in detail.groupby(group_col, sort=False):
        row = {
            group_col: name,
            "splits": g["split"].nunique(),
            "rows": int(g["rows"].sum()),
            "positives": int(g["positives"].sum()),
        }
        for metric in ["top10_capture", "auc", "pr_auc", "brier"]:
            vals = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * row[f"{metric}_std"] / math.sqrt(max(len(vals), 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation_and_sensitivity(splits: list[str], out: Path, seed: int, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    for split in splits:
        train, test, _ = read_split(split)
        for cfg in ablation_variants():
            ablation_rows.append(run_variant(train, test, split, seed=seed, max_rows=max_rows, n_estimators=280, **cfg))
        for cfg in sensitivity_variants():
            sensitivity_rows.append(
                run_variant(
                    train,
                    test,
                    split,
                    seed=seed,
                    max_rows=max_rows,
                    max_proto=cfg["max_proto"],
                    neighbors=cfg["neighbors"],
                    simulation_multiplier=cfg["simulation_multiplier"],
                    variant=cfg["variant"],
                    use_prototypes=True,
                    use_fuzzy=True,
                    use_simulation=True,
                    use_tree=True,
                    use_calibration=True,
                    n_estimators=260,
                )
            )
    ablation = pd.DataFrame(ablation_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    ablation.to_csv(out / "sds_kb_ablation_detail.csv", index=False)
    sensitivity.to_csv(out / "sds_kb_sensitivity_detail.csv", index=False)
    summarize(ablation).to_csv(out / "sds_kb_ablation_summary.csv", index=False)
    summarize(sensitivity).to_csv(out / "sds_kb_sensitivity_summary.csv", index=False)
    return ablation, sensitivity


def run_carrier_holdout(splits: list[str], out: Path, seed: int, max_rows: int) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        train, test, _ = read_split(split)
        carriers = sorted(set(train["carrier"]).intersection(set(test["carrier"])))
        for carrier in carriers:
            train_fit = train[train["carrier"].ne(carrier)].copy()
            test_hold = test[test["carrier"].eq(carrier)].copy()
            if train_fit.empty or test_hold.empty:
                continue
            row = run_variant(
                train_fit,
                test_hold,
                split,
                variant="SDS-KB carrier holdout",
                seed=seed,
                max_rows=max_rows,
                max_proto=3000,
                neighbors=5,
                simulation_multiplier=1.0,
                use_prototypes=True,
                use_fuzzy=True,
                use_simulation=True,
                use_tree=True,
                use_calibration=True,
                n_estimators=240,
            )
            row["carrier_holdout"] = carrier
            rows.append(row)
            y = test_hold["severe_late_aircraft"].to_numpy(int)
            rec = metric_dict(y, test_hold["recovery_shortfall"].to_numpy(float), None)
            rows.append({"split": split, "variant": "Recovery carrier holdout", "carrier_holdout": carrier, **rec})
    result = pd.DataFrame(rows)
    result.to_csv(out / "sds_kb_carrier_holdout_detail.csv", index=False)
    summarize(result, group_col="variant").to_csv(out / "sds_kb_carrier_holdout_summary.csv", index=False)
    return result


def read_year_panel(path: Path) -> pd.DataFrame:
    usecols = [
        "Year",
        "Month",
        "ym",
        "carrier",
        "tail",
        "origin",
        "dest",
        "prev_origin",
        "prev_dest",
        "prev_arr_delay",
        "prev_late_aircraft_delay",
        "scheduled_turnaround",
        "actual_turnaround",
        "turnaround_slack",
        "recovery_shortfall",
        "origin_hour_chain_density",
        "LateAircraftDelay",
        "severe_late_aircraft",
        *FUZZY_COLS,
        "mu_fepl_path",
    ]
    return pd.read_csv(path, usecols=usecols)


def run_cross_year(out: Path, seed: int, max_rows: int) -> pd.DataFrame:
    train = read_year_panel(YEAR_PANEL_DIR / "fepl_paths_2024_all_AA-DL-OO-UA-WN_yearonly.csv")
    test = read_year_panel(YEAR_PANEL_DIR / "fepl_paths_2025_all_AA-DL-OO-UA-WN_prevdec.csv")
    row = run_variant(
        train,
        test,
        "train2024_test2025",
        variant="SDS-KB 2024-to-2025",
        seed=seed,
        max_rows=max_rows,
        max_proto=4500,
        neighbors=5,
        simulation_multiplier=1.0,
        use_prototypes=True,
        use_fuzzy=True,
        use_simulation=True,
        use_tree=True,
        use_calibration=True,
        n_estimators=280,
    )
    y = test["severe_late_aircraft"].to_numpy(int)
    rows = [
        row,
        {"split": "train2024_test2025", "variant": "Recovery 2024-to-2025", **metric_dict(y, test["recovery_shortfall"].to_numpy(float), None)},
        {"split": "train2024_test2025", "variant": "Raw fuzzy membership 2024-to-2025", **metric_dict(y, test["mu_fepl_path"].to_numpy(float), test["mu_fepl_path"].to_numpy(float))},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(out / "sds_kb_cross_year_summary.csv", index=False)
    return result


def active_concepts(row: pd.Series) -> str:
    pairs = [
        ("buffer failure", row.get("mu_buffer_failure_path", 0.0)),
        ("late recurrence", row.get("mu_late_recurrence_path", 0.0)),
        ("congested carryover", row.get("mu_congested_carryover_path", 0.0)),
        ("upstream delay", row.get("mu_upstream_delay", 0.0)),
        ("recovery shortfall", row.get("mu_recovery_shortfall", 0.0)),
        ("tight turnaround", row.get("mu_tight_turnaround", 0.0)),
    ]
    active = [name for name, val in pairs if float(val) >= 0.5]
    return "; ".join(active) if active else "low-intensity concepts"


def run_case_evidence(split: str, out: Path, seed: int, max_rows: int) -> pd.DataFrame:
    train, test, _ = read_split(split)
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
    model = fit_lgbm(fit_aug[cols].fillna(0.0), fit_aug["severe_late_aircraft"].to_numpy(int), seed, n_estimators=280)
    cal_score = model.predict_proba(cal_df[cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test_kb[cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, cal_df["severe_late_aircraft"].to_numpy(int), test_score)
    assert info is not None
    test_kb = test_kb.copy()
    test_kb["sds_kb_score"] = test_score
    test_kb["sds_kb_risk"] = test_prob
    order = np.argsort(-test_score)
    selected = []
    severe_first = [idx for idx in order if int(test_kb.iloc[idx]["severe_late_aircraft"]) == 1][:2]
    high_risk = list(order[:20])
    for idx in [*severe_first, *high_risk]:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= 3:
            break
    rows: list[dict] = []
    case_cols = available(test_kb, CASE_COLS)
    train_case_cols = available(train, CASE_COLS)
    for rank, idx in enumerate(selected, start=1):
        path = test_kb.iloc[idx]
        sev_proto = train.iloc[int(info["test_nearest_severe_train_idx"][idx])]
        norm_proto = train.iloc[int(info["test_nearest_normal_train_idx"][idx])]
        row = {
            "case": f"{split}_case{rank}",
            "split": split,
            "carrier": path.get("carrier"),
            "path": f"{path.get('prev_origin')}->{path.get('origin')}->{path.get('dest')}",
            "scheduled_turnaround": float(path.get("scheduled_turnaround", np.nan)),
            "prev_arr_delay": float(path.get("prev_arr_delay", np.nan)),
            "recovery_shortfall": float(path.get("recovery_shortfall", np.nan)),
            "late_aircraft_delay": float(path.get("LateAircraftDelay", np.nan)),
            "severe_label": int(path.get("severe_late_aircraft", 0)),
            "sds_kb_score": float(path["sds_kb_score"]),
            "calibrated_risk": float(path["sds_kb_risk"]),
            "severe_distance": float(info["test_nearest_severe_dist"][idx]),
            "normal_distance": float(info["test_nearest_normal_dist"][idx]),
            "distance_margin": float(path["kb_distance_margin"]),
            "active_concepts": active_concepts(path),
            "nearest_severe_path": f"{sev_proto.get('prev_origin')}->{sev_proto.get('origin')}->{sev_proto.get('dest')}",
            "nearest_severe_late_aircraft_delay": float(sev_proto.get("LateAircraftDelay", np.nan)),
            "nearest_severe_recovery_shortfall": float(sev_proto.get("recovery_shortfall", np.nan)),
            "nearest_normal_path": f"{norm_proto.get('prev_origin')}->{norm_proto.get('origin')}->{norm_proto.get('dest')}",
            "nearest_normal_late_aircraft_delay": float(norm_proto.get("LateAircraftDelay", np.nan)),
            "nearest_normal_recovery_shortfall": float(norm_proto.get("recovery_shortfall", np.nan)),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "sds_kb_audit_cases.csv", index=False)
    return result


def write_report(out: Path) -> None:
    lines = ["# SDS-KB strengthening validation", ""]
    for filename, title in [
        ("sds_kb_ablation_summary.csv", "Component ablation"),
        ("sds_kb_sensitivity_summary.csv", "Parameter sensitivity"),
        ("sds_kb_carrier_holdout_summary.csv", "Carrier holdout"),
        ("sds_kb_cross_year_summary.csv", "Cross-year transfer"),
        ("sds_kb_audit_cases.csv", "Knowledge-base audit cases"),
    ]:
        path = out / filename
        if path.exists():
            df = pd.read_csv(path)
            lines.extend([f"## {title}", "", df.to_markdown(index=False), ""])
    (out / "sds_kb_strengthening_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="2025_01_to_2025_02,2025_07_to_2025_08")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--run-carrier", action="store_true")
    parser.add_argument("--run-cross-year", action="store_true")
    parser.add_argument("--case-split", default="2025_07_to_2025_08")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    ablation, sensitivity = run_ablation_and_sensitivity(splits, out, args.seed, args.max_train_rows)
    print("Ablation summary")
    print(summarize(ablation).to_string(index=False))
    print("Sensitivity summary")
    print(summarize(sensitivity).to_string(index=False))
    run_case_evidence(args.case_split, out, args.seed, args.max_train_rows)
    if args.run_carrier:
        carrier = run_carrier_holdout(splits, out, args.seed, max(50000, args.max_train_rows // 2))
        print("Carrier holdout summary")
        print(summarize(carrier, group_col="variant").to_string(index=False))
    if args.run_cross_year:
        cross = run_cross_year(out, args.seed, max(120000, args.max_train_rows))
        print("Cross-year summary")
        print(cross.to_string(index=False))
    write_report(out)


if __name__ == "__main__":
    main()

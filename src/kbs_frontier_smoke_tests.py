from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT / "results" / "full_fepl_validation" / "splits"
OUT_DEFAULT = PROJECT / "results" / "kbs_frontier_smoke_tests"

CORE_COLS = [
    "ym",
    "FlightDate",
    "carrier",
    "tail",
    "origin",
    "dest",
    "prev_origin",
    "prev_dest",
    "prev_carrier",
    "prev_arr_delay",
    "prev_dep_delay",
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
    "LateAircraftDelay",
    "severe_late_aircraft",
    "mu_upstream_delay",
    "mu_prev_late_aircraft",
    "mu_tight_turnaround",
    "mu_recovery_shortfall",
    "mu_origin_chain_density",
    "mu_buffer_failure_path",
    "mu_late_recurrence_path",
    "mu_congested_carryover_path",
    "mu_fepl_path",
]

BASE_NUMERIC = [
    "prev_arr_delay",
    "prev_late_aircraft_delay",
    "scheduled_turnaround",
    "actual_turnaround",
    "turnaround_slack",
    "recovery_shortfall",
    "crs_elapsed",
    "distance",
    "dep_hour",
    "origin_hour_chain_density",
    "mu_upstream_delay",
    "mu_prev_late_aircraft",
    "mu_tight_turnaround",
    "mu_recovery_shortfall",
    "mu_origin_chain_density",
    "mu_buffer_failure_path",
    "mu_late_recurrence_path",
    "mu_congested_carryover_path",
]

BOUNDARY_CANDIDATES = {
    "upstream_delay": "mu_upstream_delay",
    "recovery_shortfall": "mu_recovery_shortfall",
    "previous_late_aircraft": "mu_prev_late_aircraft",
    "tight_turnaround": "mu_tight_turnaround",
    "rotation_density": "mu_origin_chain_density",
    "buffer_failure": "mu_buffer_failure_path",
    "late_recurrence": "mu_late_recurrence_path",
    "congested_carryover": "mu_congested_carryover_path",
    "actual_turnaround": "actual_turnaround",
    "scheduled_turnaround": "scheduled_turnaround",
}


def read_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    folder = SPLIT_DIR / split
    metadata = json.loads((folder / "split_metadata.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(folder / "fepl_rotation_path_panel.csv", usecols=CORE_COLS)
    train = panel[panel["ym"].eq(metadata["train_month"])].copy()
    test = panel[panel["ym"].eq(metadata["test_month"])].copy()
    return train, test, metadata


def top_capture(y: np.ndarray, score: np.ndarray, share: float = 0.10) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(share * len(y))))
    order = np.argpartition(-score, k - 1)[:k] if k < len(y) else np.arange(len(y))
    positives = max(int(y.sum()), 1)
    return float(y[order].sum() / positives), float(y[order].mean())


def metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
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


def stratified_sample(train: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = train[train["severe_late_aircraft"].eq(1)]
    neg = train[train["severe_late_aircraft"].eq(0)]
    pos_keep = min(len(pos), max(1, int(max_rows * 0.35)))
    neg_keep = max_rows - pos_keep
    return pd.concat(
        [
            pos.sample(pos_keep, random_state=seed),
            neg.sample(min(len(neg), neg_keep), random_state=seed),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def fit_lgbm(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, n_estimators: int = 260) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=seed,
        class_weight="balanced",
        n_jobs=4,
        verbosity=-1,
    ).fit(x_train, y_train)


def calibrate(train_score: np.ndarray, y_train: np.ndarray, test_score: np.ndarray) -> np.ndarray:
    if np.max(train_score) <= np.min(train_score):
        return np.full(len(test_score), float(np.mean(y_train)))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(train_score, dtype=float), np.asarray(y_train, dtype=int))
    return np.clip(iso.predict(np.asarray(test_score, dtype=float)), 0.0, 1.0)


def split_fit_calibration(train: pd.DataFrame, seed: int, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = stratified_sample(train, max_rows, seed)
    sort_cols = [col for col in ["FlightDate", "Year", "Month", "ym", "carrier", "origin", "dest"] if col in sample.columns]
    sample = sample.sort_values(sort_cols).reset_index(drop=True)
    cut = max(1, int(0.78 * len(sample)))
    return sample.iloc[:cut].copy(), sample.iloc[cut:].copy()


def evaluate_model(
    route: str,
    split: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    concept_count: int,
    seed: int,
    max_rows: int,
    n_estimators: int = 260,
) -> tuple[dict, np.ndarray, np.ndarray]:
    fit_df, cal_df = split_fit_calibration(train, seed, max_rows)
    y_fit = fit_df["severe_late_aircraft"].to_numpy(int)
    y_cal = cal_df["severe_late_aircraft"].to_numpy(int)
    y_test = test["severe_late_aircraft"].to_numpy(int)
    model = fit_lgbm(fit_df[feature_cols].fillna(0.0), y_fit, seed, n_estimators)
    cal_score = model.predict_proba(cal_df[feature_cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test[feature_cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, y_cal, test_score)
    row = {
        "route": route,
        "split": split,
        "concept_count": concept_count,
        "feature_count": len(feature_cols),
        **metrics(y_test, test_score, test_prob),
    }
    return row, test_score, test_prob


def route1_causal_boundary(train: pd.DataFrame, test: pd.DataFrame, split: str, seed: int, max_rows: int) -> tuple[dict, list[str]]:
    fit_df, val_df = split_fit_calibration(train, seed, max_rows)
    y_fit = fit_df["severe_late_aircraft"].to_numpy(int)
    y_val = val_df["severe_late_aircraft"].to_numpy(int)
    selected: list[str] = []
    remaining = list(BOUNDARY_CANDIDATES)
    best_score = -np.inf
    for _ in range(5):
        best_candidate = None
        for concept in remaining:
            concepts = selected + [concept]
            cols = [BOUNDARY_CANDIDATES[c] for c in concepts]
            model = fit_lgbm(fit_df[cols].fillna(0.0), y_fit, seed, n_estimators=140)
            score = model.predict_proba(val_df[cols].fillna(0.0))[:, 1]
            t10, _ = top_capture(y_val, score)
            pr_auc = average_precision_score(y_val, score)
            utility = pr_auc + 0.15 * t10 - 0.006 * len(concepts)
            if best_candidate is None or utility > best_candidate[0]:
                best_candidate = (utility, concept)
        if best_candidate is None or best_candidate[0] <= best_score + 1e-4:
            break
        best_score, concept = best_candidate
        selected.append(concept)
        remaining.remove(concept)
    if not selected:
        selected = ["buffer_failure"]
    cols = [BOUNDARY_CANDIDATES[c] for c in selected]
    row, _, _ = evaluate_model(
        "R1 rotation causal boundary",
        split,
        train,
        test,
        cols,
        concept_count=len(selected),
        seed=seed,
        max_rows=max_rows,
        n_estimators=260,
    )
    row["selected_concepts"] = "; ".join(selected)
    return row, selected


def entropy(p: pd.Series) -> pd.Series:
    p = p.clip(1e-6, 1.0 - 1e-6)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def add_entropy_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    for df in [train, test]:
        df["turn_bin"] = pd.cut(df["scheduled_turnaround"], [-1, 45, 70, 100, 140, 9999], labels=False).fillna(4).astype(int)
        df["prevlate_bin"] = pd.cut(df["prev_late_aircraft_delay"], [-1, 0, 15, 45, 90, 9999], labels=False).fillna(0).astype(int)
        df["shortfall_bin"] = pd.cut(df["recovery_shortfall"], [-1, 0, 15, 45, 90, 9999], labels=False).fillna(0).astype(int)
        df["hour_bin"] = (pd.to_numeric(df["dep_hour"], errors="coerce").fillna(0).astype(int) // 4).clip(0, 5)
    keys = {
        "carrier_origin_hour": ["carrier", "origin", "hour_bin"],
        "route_hour": ["origin", "dest", "hour_bin"],
        "carrier_route": ["carrier", "origin", "dest"],
        "turn_prevlate": ["turn_bin", "prevlate_bin"],
        "turn_shortfall": ["turn_bin", "shortfall_bin"],
        "airport_turn": ["origin", "turn_bin"],
    }
    prior = float(train["severe_late_aircraft"].mean())
    new_cols: list[str] = []
    for name, group_cols in keys.items():
        stats = train.groupby(group_cols)["severe_late_aircraft"].agg(["mean", "size"]).reset_index()
        stats[f"{name}_rate"] = (stats["mean"] * stats["size"] + prior * 40.0) / (stats["size"] + 40.0)
        stats[f"{name}_entropy"] = entropy(stats[f"{name}_rate"])
        stats[f"{name}_reliability"] = (1.0 - stats[f"{name}_entropy"]) * np.log1p(stats["size"])
        keep = [*group_cols, f"{name}_rate", f"{name}_entropy", f"{name}_reliability"]
        stats = stats[keep]
        for df in [train, test]:
            merged = df.merge(stats, on=group_cols, how="left")
            df[f"{name}_rate"] = merged[f"{name}_rate"].fillna(prior).to_numpy()
            df[f"{name}_entropy"] = merged[f"{name}_entropy"].fillna(entropy(pd.Series([prior])).iloc[0]).to_numpy()
            df[f"{name}_reliability"] = merged[f"{name}_reliability"].fillna(0.0).to_numpy()
        new_cols.extend([f"{name}_rate", f"{name}_entropy", f"{name}_reliability"])
    return train, test, new_cols


def route2_entropy_graph(train: pd.DataFrame, test: pd.DataFrame, split: str, seed: int, max_rows: int) -> dict:
    train_g, test_g, graph_cols = add_entropy_features(train, test)
    cols = [
        "mu_buffer_failure_path",
        "mu_late_recurrence_path",
        "mu_congested_carryover_path",
        "recovery_shortfall",
        "prev_arr_delay",
        *graph_cols,
    ]
    row, _, _ = evaluate_model(
        "R2 entropy-gated rotation knowledge graph",
        split,
        train_g,
        test_g,
        cols,
        concept_count=6,
        seed=seed,
        max_rows=max_rows,
        n_estimators=260,
    )
    row["selected_concepts"] = "six entropy-gated group families"
    return row


def add_prototype_features(train: pd.DataFrame, test: pd.DataFrame, seed: int, max_proto: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    proto_cols = [
        "prev_arr_delay",
        "prev_late_aircraft_delay",
        "scheduled_turnaround",
        "actual_turnaround",
        "recovery_shortfall",
        "origin_hour_chain_density",
        "mu_buffer_failure_path",
        "mu_late_recurrence_path",
        "mu_congested_carryover_path",
    ]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[proto_cols].fillna(0.0).to_numpy(float))
    test_x = scaler.transform(test[proto_cols].fillna(0.0).to_numpy(float))
    severe_idx = np.where(train["severe_late_aircraft"].to_numpy(int) == 1)[0]
    normal_idx = np.where(train["severe_late_aircraft"].to_numpy(int) == 0)[0]
    rng = np.random.default_rng(seed)
    severe_idx = rng.choice(severe_idx, size=min(len(severe_idx), max_proto), replace=False)
    normal_idx = rng.choice(normal_idx, size=min(len(normal_idx), max_proto), replace=False)
    severe_nn = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(train_x[severe_idx])
    normal_nn = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(train_x[normal_idx])
    train_d_sev, _ = severe_nn.kneighbors(train_x)
    train_d_norm, _ = normal_nn.kneighbors(train_x)
    test_d_sev, _ = severe_nn.kneighbors(test_x)
    test_d_norm, _ = normal_nn.kneighbors(test_x)
    train = train.copy()
    test = test.copy()
    for df, d_sev, d_norm in [(train, train_d_sev, train_d_norm), (test, test_d_sev, test_d_norm)]:
        df["kb_dist_severe"] = d_sev.mean(axis=1)
        df["kb_dist_normal"] = d_norm.mean(axis=1)
        df["kb_distance_margin"] = df["kb_dist_normal"] - df["kb_dist_severe"]
        df["kb_severe_affinity"] = 1.0 / (1.0 + df["kb_dist_severe"])
        df["kb_normal_affinity"] = 1.0 / (1.0 + df["kb_dist_normal"])
    return train, test, ["kb_dist_severe", "kb_dist_normal", "kb_distance_margin", "kb_severe_affinity", "kb_normal_affinity"]


def augment_severe(train: pd.DataFrame, feature_cols: list[str], seed: int, multiplier: float) -> pd.DataFrame:
    severe = train[train["severe_late_aircraft"].eq(1)].copy()
    if severe.empty or multiplier <= 0:
        return train
    rng = np.random.default_rng(seed)
    keep = severe.sample(int(len(severe) * multiplier), replace=True, random_state=seed).copy()
    for col in feature_cols:
        arr = pd.to_numeric(severe[col], errors="coerce").fillna(0.0).to_numpy(float)
        scale = max(float(np.nanstd(arr)), 1.0) * 0.035
        keep[col] = pd.to_numeric(keep[col], errors="coerce").fillna(0.0) + rng.normal(0.0, scale, size=len(keep))
    keep["severe_late_aircraft"] = 1
    return pd.concat([train, keep], ignore_index=True)


def route5_simulation_kb(train: pd.DataFrame, test: pd.DataFrame, split: str, seed: int, max_rows: int) -> dict:
    train_kb, test_kb, kb_cols = add_prototype_features(train, test, seed, max_proto=4500)
    cols = [*BASE_NUMERIC, *kb_cols]
    fit_df, cal_df = split_fit_calibration(train_kb, seed, max_rows)
    fit_aug = augment_severe(fit_df, cols, seed, multiplier=1.0)
    y_fit = fit_aug["severe_late_aircraft"].to_numpy(int)
    y_cal = cal_df["severe_late_aircraft"].to_numpy(int)
    y_test = test_kb["severe_late_aircraft"].to_numpy(int)
    model = fit_lgbm(fit_aug[cols].fillna(0.0), y_fit, seed, n_estimators=300)
    cal_score = model.predict_proba(cal_df[cols].fillna(0.0))[:, 1]
    test_score = model.predict_proba(test_kb[cols].fillna(0.0))[:, 1]
    test_prob = calibrate(cal_score, y_cal, test_score)
    row = {
        "route": "R5 severe-delay simulation knowledge base",
        "split": split,
        "concept_count": 3,
        "feature_count": len(cols),
        "selected_concepts": "severe prototype; normal prototype; simulated severe path",
        **metrics(y_test, test_score, test_prob),
    }
    return row


def add_state_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    def state(df: pd.DataFrame) -> pd.Series:
        up = df["mu_upstream_delay"].to_numpy(float)
        short = df["mu_recovery_shortfall"].to_numpy(float)
        prev = df["mu_prev_late_aircraft"].to_numpy(float)
        tight = df["mu_tight_turnaround"].to_numpy(float)
        states = np.full(len(df), 0)
        states[(up >= 0.5) & (short < 0.35)] = 1
        states[(short >= 0.5) & (prev < 0.35)] = 2
        states[(up >= 0.5) & (short >= 0.5)] = 3
        states[(prev >= 0.5) & ((short >= 0.35) | (tight >= 0.5))] = 4
        return pd.Series(states, index=df.index)

    train = train.copy()
    test = test.copy()
    train["auto_state"] = state(train)
    test["auto_state"] = state(test)
    prior = float(train["severe_late_aircraft"].mean())
    stats = train.groupby("auto_state")["severe_late_aircraft"].agg(["mean", "size"]).reset_index()
    stats["state_rate"] = (stats["mean"] * stats["size"] + prior * 40.0) / (stats["size"] + 40.0)
    stats["state_conf"] = np.log1p(stats["size"])
    for df in [train, test]:
        merged = df.merge(stats[["auto_state", "state_rate", "state_conf"]], on="auto_state", how="left")
        df["state_rate"] = merged["state_rate"].fillna(prior).to_numpy()
        df["state_conf"] = merged["state_conf"].fillna(0.0).to_numpy()
    return train, test, ["auto_state", "state_rate", "state_conf"]


def route4_automaton(train: pd.DataFrame, test: pd.DataFrame, split: str, seed: int, max_rows: int) -> dict:
    train_s, test_s, state_cols = add_state_features(train, test)
    cols = [
        "mu_buffer_failure_path",
        "mu_late_recurrence_path",
        "mu_congested_carryover_path",
        "mu_recovery_shortfall",
        *state_cols,
    ]
    row, _, _ = evaluate_model(
        "R4 prototype transition automaton",
        split,
        train_s,
        test_s,
        cols,
        concept_count=5,
        seed=seed,
        max_rows=max_rows,
        n_estimators=220,
    )
    row["selected_concepts"] = "Normal; Recovered; Fragile; Propagating; Attribution-confirmed"
    return row


def baseline_rows(split: str, test: pd.DataFrame) -> list[dict]:
    y = test["severe_late_aircraft"].to_numpy(int)
    return [
        {"route": "Current FEPL reference", "split": split, "concept_count": 2, "feature_count": 3, "selected_concepts": "existing FEPL memberships", **metrics(y, test["mu_fepl_path"].to_numpy(float), test["mu_fepl_path"].to_numpy(float))},
        {"route": "Recovery shortfall reference", "split": split, "concept_count": 1, "feature_count": 1, "selected_concepts": "recovery shortfall", **metrics(y, test["recovery_shortfall"].to_numpy(float), None)},
    ]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby("route", as_index=False).agg(
        splits=("split", "nunique"),
        rows=("rows", "sum"),
        positives=("positives", "sum"),
        mean_top10_capture=("top10_capture", "mean"),
        mean_auc=("auc", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_brier=("brier", "mean"),
        mean_concept_count=("concept_count", "mean"),
        mean_feature_count=("feature_count", "mean"),
    )
    summary["passes_pr_088"] = summary["mean_pr_auc"].ge(0.88)
    summary["passes_pr_085_t10_0965"] = summary["mean_pr_auc"].ge(0.85) & summary["mean_top10_capture"].ge(0.965)
    summary["passes_action_t10_097"] = summary["mean_top10_capture"].ge(0.97)
    return summary.sort_values(["passes_pr_088", "mean_pr_auc", "mean_top10_capture"], ascending=False)


def write_report(out: Path, detail: pd.DataFrame, summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# KBS Frontier Smoke Tests",
        "",
        "Smoke tests use Jan-to-Feb and Jul-to-Aug 2025 splits from the cached aircraft-rotation path panels.",
        "Higher Top-10% capture, AUC, and PR-AUC are better. Lower Brier score is better.",
        "",
        "## Route Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Selected Concepts",
        "",
        selected.to_markdown(index=False),
        "",
        "## Split Detail",
        "",
        detail.to_markdown(index=False),
        "",
    ]
    (out / "kbs_frontier_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def route1_with_fixed_concepts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    split: str,
    seed: int,
    max_rows: int,
    concepts: list[str],
    route_name: str,
) -> dict:
    cols = [BOUNDARY_CANDIDATES[c] for c in concepts]
    row, _, _ = evaluate_model(
        route_name,
        split,
        train,
        test,
        cols,
        concept_count=len(concepts),
        seed=seed,
        max_rows=max_rows,
        n_estimators=240,
    )
    row["selected_concepts"] = "; ".join(concepts)
    return row


def route5_fixed_holdout(train: pd.DataFrame, test: pd.DataFrame, split: str, seed: int, max_rows: int, route_name: str) -> dict:
    return route5_simulation_kb(train, test, split, seed, max_rows) | {"route": route_name}


def airport_holdout_smoke(splits: list[str], seed: int, max_rows: int, out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        train, test, _ = read_split(split)
        heldout_airports = set(test["origin"].value_counts().head(20).index)
        train_fit = train[~train["origin"].isin(heldout_airports)].copy()
        test_hold = test[test["origin"].isin(heldout_airports)].copy()
        if train_fit.empty or test_hold.empty:
            continue
        rows.append(
            route1_with_fixed_concepts(
                train_fit,
                test_hold,
                split,
                seed,
                max_rows,
                ["buffer_failure", "actual_turnaround", "scheduled_turnaround"],
                "R1 airport holdout",
            )
        )
        rows.append(route5_fixed_holdout(train_fit, test_hold, split, seed, max_rows, "R5 airport holdout"))
        y = test_hold["severe_late_aircraft"].to_numpy(int)
        rows.append(
            {
                "route": "Recovery shortfall airport reference",
                "split": split,
                "concept_count": 1,
                "feature_count": 1,
                "selected_concepts": "recovery shortfall",
                **metrics(y, test_hold["recovery_shortfall"].to_numpy(float), None),
            }
        )
    holdout = pd.DataFrame(rows)
    if holdout.empty:
        return holdout
    holdout.to_csv(out / "kbs_frontier_airport_holdout_detail.csv", index=False)
    summary = summarize(holdout)
    summary.to_csv(out / "kbs_frontier_airport_holdout_summary.csv", index=False)
    return holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="2025_01_to_2025_02,2025_07_to_2025_08")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--routes", default="1,2,5,4")
    parser.add_argument("--skip-airport-holdout", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    route_ids = {item.strip() for item in args.routes.split(",") if item.strip()}
    rows: list[dict] = []
    selected_rows: list[dict] = []
    for split in splits:
        train, test, metadata = read_split(split)
        rows.extend(baseline_rows(split, test))
        if "1" in route_ids:
            row, selected = route1_causal_boundary(train, test, split, args.seed, args.max_train_rows)
            rows.append(row)
            selected_rows.append({"route": row["route"], "split": split, "selected": "; ".join(selected)})
        if "2" in route_ids:
            row = route2_entropy_graph(train, test, split, args.seed, args.max_train_rows)
            rows.append(row)
            selected_rows.append({"route": row["route"], "split": split, "selected": row["selected_concepts"]})
        if "5" in route_ids:
            row = route5_simulation_kb(train, test, split, args.seed, args.max_train_rows)
            rows.append(row)
            selected_rows.append({"route": row["route"], "split": split, "selected": row["selected_concepts"]})
        if "4" in route_ids:
            row = route4_automaton(train, test, split, args.seed, args.max_train_rows)
            rows.append(row)
            selected_rows.append({"route": row["route"], "split": split, "selected": row["selected_concepts"]})

    detail = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    summary = summarize(detail)
    detail.to_csv(args.out / "kbs_frontier_smoke_detail.csv", index=False)
    summary.to_csv(args.out / "kbs_frontier_smoke_summary.csv", index=False)
    selected.to_csv(args.out / "kbs_frontier_selected_concepts.csv", index=False)
    holdout = pd.DataFrame()
    if not args.skip_airport_holdout:
        holdout = airport_holdout_smoke(splits, args.seed, args.max_train_rows, args.out)
    write_report(args.out, detail, summary, selected)
    if not holdout.empty:
        print("\nAirport holdout smoke:")
        print(summarize(holdout).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

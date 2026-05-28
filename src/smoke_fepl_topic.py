from __future__ import annotations

import argparse
import itertools
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw_open" / "bts_on_time"
OUT_DEFAULT = PROJECT / "results" / "smoke_fepl_topic"

USECOLS = [
    "Year",
    "Month",
    "DayofMonth",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelayMinutes",
    "CRSArrTime",
    "ArrTime",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

NUMERIC_FEATURES = [
    "prev_arr_delay",
    "prev_dep_delay",
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
    "mu_fepl_path",
]

CATEGORICAL_FEATURES = ["carrier", "origin", "dest"]


def month_zip(year: int, month: int) -> Path:
    return RAW / f"bts_on_time_{year}_{month:02d}.zip"


def read_bts_month(year: int, month: int, carriers: set[str] | None) -> pd.DataFrame:
    path = month_zip(year, month)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        csv_name = [name for name in z.namelist() if name.lower().endswith(".csv")][0]
        df = pd.read_csv(z.open(csv_name), usecols=USECOLS, low_memory=False)
    df = df.rename(
        columns={
            "Reporting_Airline": "carrier",
            "Tail_Number": "tail",
            "Origin": "origin",
            "Dest": "dest",
            "CRSElapsedTime": "crs_elapsed",
            "ActualElapsedTime": "actual_elapsed",
            "Distance": "distance",
        }
    )
    if carriers:
        df = df[df["carrier"].isin(carriers)].copy()
    return df


def hhmm_to_minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(".0", "")
    if not text:
        return np.nan
    try:
        raw = int(float(text))
    except ValueError:
        return np.nan
    hour = raw // 100
    minute = raw % 100
    if hour == 24:
        hour = 0
    if hour < 0 or hour > 24 or minute < 0 or minute >= 60:
        return np.nan
    return float(hour * 60 + minute)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flight_date"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    for col in ["CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"]:
        df[f"{col}_min"] = df[col].map(hhmm_to_minutes)
    df = df.dropna(subset=["flight_date", "DepTime_min", "ArrTime_min", "CRSDepTime_min", "CRSArrTime_min", "tail"])
    df["dep_dt"] = df["flight_date"] + pd.to_timedelta(df["DepTime_min"], unit="m")
    df["arr_dt"] = df["flight_date"] + pd.to_timedelta(df["ArrTime_min"], unit="m")
    df.loc[df["arr_dt"] < df["dep_dt"], "arr_dt"] += pd.Timedelta(days=1)
    df["crs_dep_dt"] = df["flight_date"] + pd.to_timedelta(df["CRSDepTime_min"], unit="m")
    df["crs_arr_dt"] = df["flight_date"] + pd.to_timedelta(df["CRSArrTime_min"], unit="m")
    df.loc[df["crs_arr_dt"] < df["crs_dep_dt"], "crs_arr_dt"] += pd.Timedelta(days=1)
    df["dep_hour"] = (df["CRSDepTime_min"] // 60).clip(0, 23).astype(int)
    return df


def ramp_up(values: pd.Series, low: float, high: float) -> pd.Series:
    return ((pd.to_numeric(values, errors="coerce").fillna(0.0) - low) / (high - low)).clip(0.0, 1.0)


def ramp_down(values: pd.Series, low: float, high: float) -> pd.Series:
    return (1.0 - (pd.to_numeric(values, errors="coerce").fillna(high) - low) / (high - low)).clip(0.0, 1.0)


def build_rotation_paths(months: list[tuple[int, int]], carriers: set[str], out: Path) -> pd.DataFrame:
    frames = [read_bts_month(year, month, carriers) for year, month in months]
    flights = pd.concat(frames, ignore_index=True)
    flights = flights[
        (pd.to_numeric(flights["Cancelled"], errors="coerce").fillna(0).eq(0))
        & (pd.to_numeric(flights["Diverted"], errors="coerce").fillna(0).eq(0))
    ].copy()
    flights = add_time_columns(flights)
    numeric_cols = [
        "DepDelayMinutes",
        "ArrDelayMinutes",
        "LateAircraftDelay",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "crs_elapsed",
        "actual_elapsed",
        "distance",
    ]
    for col in numeric_cols:
        flights[col] = pd.to_numeric(flights[col], errors="coerce").fillna(0.0)

    flights = flights.sort_values(["tail", "dep_dt", "arr_dt"]).reset_index(drop=True)
    prev_cols = {
        "tail": "prev_tail",
        "carrier": "prev_carrier",
        "origin": "prev_origin",
        "dest": "prev_dest",
        "dep_dt": "prev_dep_dt",
        "arr_dt": "prev_arr_dt",
        "crs_dep_dt": "prev_crs_dep_dt",
        "crs_arr_dt": "prev_crs_arr_dt",
        "DepDelayMinutes": "prev_dep_delay",
        "ArrDelayMinutes": "prev_arr_delay",
        "LateAircraftDelay": "prev_late_aircraft_delay",
    }
    prev = flights[list(prev_cols)].rename(columns=prev_cols).shift(1)
    paths = pd.concat([flights, prev], axis=1)
    paths = paths[
        paths["tail"].eq(paths["prev_tail"])
        & paths["prev_dest"].eq(paths["origin"])
        & paths["prev_arr_dt"].notna()
    ].copy()
    paths["actual_turnaround"] = (paths["dep_dt"] - paths["prev_arr_dt"]).dt.total_seconds() / 60.0
    paths["scheduled_turnaround"] = (paths["crs_dep_dt"] - paths["prev_crs_arr_dt"]).dt.total_seconds() / 60.0
    paths = paths[
        paths["actual_turnaround"].between(20, 720)
        & paths["scheduled_turnaround"].between(15, 720)
    ].copy()
    paths["turnaround_slack"] = (paths["scheduled_turnaround"] - 35.0).clip(lower=0.0)
    paths["recovery_shortfall"] = (paths["prev_arr_delay"] - paths["turnaround_slack"]).clip(lower=0.0)
    paths["severe_late_aircraft"] = paths["LateAircraftDelay"].ge(30.0).astype(int)
    paths["any_late_aircraft"] = paths["LateAircraftDelay"].gt(0.0).astype(int)
    paths["ym"] = paths["Year"].astype(int).astype(str) + "_" + paths["Month"].astype(int).astype(str).str.zfill(2)

    density = (
        paths.groupby(["ym", "origin", "dep_hour"], as_index=False)
        .size()
        .rename(columns={"size": "origin_hour_chain_density"})
    )
    paths = paths.merge(density, on=["ym", "origin", "dep_hour"], how="left")
    paths["mu_upstream_delay"] = ramp_up(paths["prev_arr_delay"], 15.0, 90.0)
    paths["mu_prev_late_aircraft"] = ramp_up(paths["prev_late_aircraft_delay"], 10.0, 75.0)
    paths["mu_tight_turnaround"] = ramp_down(paths["scheduled_turnaround"], 45.0, 140.0)
    paths["mu_recovery_shortfall"] = ramp_up(paths["recovery_shortfall"], 0.0, 75.0)
    q_density = paths["origin_hour_chain_density"].quantile(0.95)
    q_density = q_density if np.isfinite(q_density) and q_density > 0 else 1.0
    paths["mu_origin_chain_density"] = (paths["origin_hour_chain_density"] / q_density).clip(0.0, 1.0)
    paths["mu_buffer_failure_path"] = np.maximum(
        paths["mu_upstream_delay"] * paths["mu_tight_turnaround"],
        paths["mu_recovery_shortfall"],
    )
    paths["mu_late_recurrence_path"] = paths["mu_prev_late_aircraft"] * (0.35 + 0.65 * paths["mu_tight_turnaround"])
    paths["mu_congested_carryover_path"] = paths["mu_upstream_delay"] * paths["mu_origin_chain_density"]
    paths["mu_fepl_path"] = np.maximum.reduce(
        [
            paths["mu_buffer_failure_path"].to_numpy(float),
            paths["mu_late_recurrence_path"].to_numpy(float),
            paths["mu_congested_carryover_path"].to_numpy(float),
        ]
    )
    keep = [
        "Year",
        "Month",
        "FlightDate",
        "carrier",
        "tail",
        "origin",
        "dest",
        "prev_origin",
        "prev_dest",
        "prev_carrier",
        "dep_dt",
        "arr_dt",
        "prev_arr_dt",
        "DepDelayMinutes",
        "ArrDelayMinutes",
        "LateAircraftDelay",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "prev_dep_delay",
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
        "severe_late_aircraft",
        "any_late_aircraft",
        "ym",
    ] + [col for col in paths.columns if col.startswith("mu_")]
    paths = paths[keep].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    paths.to_csv(out / "fepl_rotation_path_panel.csv", index=False)
    return paths


def top_capture(y: np.ndarray, score: np.ndarray, share: float = 0.10) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    total = y.sum()
    k = max(1, int(math.ceil(share * len(y))))
    order = np.argsort(-score)
    top = order[:k]
    capture = float(y[top].sum() / total) if total > 0 else np.nan
    lift = capture / share if total > 0 else np.nan
    hit = float(y[top].mean())
    return capture, lift, hit


def metric_dict(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prob = np.clip(np.asarray(prob if prob is not None else score, dtype=float), 0.0, 1.0)
    capture, lift, hit = top_capture(y, score)
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, score))
        pr_auc = float(average_precision_score(y, score))
    else:
        auc = np.nan
        pr_auc = np.nan
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "top10_capture": capture,
        "top10_lift": lift,
        "top10_hit_rate": hit,
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(y, prob)),
        "rmse": float(np.sqrt(np.mean((y - prob) ** 2))),
    }


def fit_fepl(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    terms = [
        ("buffer_failure", [0.0, 0.25, 0.50, 1.00, 1.50], lambda d: d["mu_buffer_failure_path"].to_numpy(float)),
        ("late_recurrence", [0.0, 0.25, 0.50, 1.00], lambda d: d["mu_late_recurrence_path"].to_numpy(float)),
        ("congested_carryover", [0.0, 0.25, 0.50, 1.00], lambda d: d["mu_congested_carryover_path"].to_numpy(float)),
        ("raw_upstream", [0.0, 0.25, 0.50], lambda d: d["mu_upstream_delay"].to_numpy(float)),
        ("shortfall", [0.0, 0.25, 0.50, 1.00], lambda d: d["mu_recovery_shortfall"].to_numpy(float)),
    ]
    y = train["severe_late_aircraft"].to_numpy(int)
    best = None
    for weights in itertools.product(*[grid for _, grid, _ in terms]):
        if not any(weight > 0 for weight in weights):
            continue
        score = np.zeros(len(train), dtype=float)
        for w, (_, _, fn) in zip(weights, terms):
            score += w * fn(train)
        metrics = metric_dict(y, score, np.clip(score / max(score.max(), 1.0), 0, 1))
        key = (metrics["top10_capture"], metrics["auc"], metrics["pr_auc"])
        if best is None or key > best[0]:
            best = (key, weights)
    weights = best[1]
    train_score = np.zeros(len(train), dtype=float)
    test_score = np.zeros(len(test), dtype=float)
    for w, (_, _, fn) in zip(weights, terms):
        train_score += w * fn(train)
        test_score += w * fn(test)
    train_rank = train_score.clip(0.0, None)
    test_rank = test_score.clip(0.0, None)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train_rank, y)
    prob = iso.predict(test_rank).clip(0.0, 1.0)
    weight_map = {name: float(w) for w, (name, _, _) in zip(weights, terms)}
    return test_rank, prob, weight_map


def sample_train(train: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train
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
    ).sample(frac=1.0, random_state=seed)


def fit_tree_models(train: pd.DataFrame, test: pd.DataFrame, seed: int, max_train_rows: int) -> dict[str, np.ndarray]:
    train_fit = sample_train(train, max_train_rows, seed)
    y = train_fit["severe_late_aircraft"].to_numpy(int)
    x_train_num = train_fit[NUMERIC_FEATURES].astype(float)
    x_test_num = test[NUMERIC_FEATURES].astype(float)
    x_train = pd.concat([x_train_num, pd.get_dummies(train_fit[CATEGORICAL_FEATURES].astype(str), dtype=float)], axis=1)
    x_test = pd.concat([x_test_num, pd.get_dummies(test[CATEGORICAL_FEATURES].astype(str), dtype=float)], axis=1)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
    scale_pos = max(float((y == 0).sum() / max((y == 1).sum(), 1)), 1.0)
    preds: dict[str, np.ndarray] = {}

    lgbm = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight="balanced",
        verbosity=-1,
    )
    lgbm.fit(x_train, y)
    preds["LightGBM path baseline"] = lgbm.predict_proba(x_test)[:, 1]

    xgb = XGBClassifier(
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
    xgb.fit(x_train, y)
    preds["XGBoost path baseline"] = xgb.predict_proba(x_test)[:, 1]

    cat_cols = CATEGORICAL_FEATURES
    cat_features = [train_fit[NUMERIC_FEATURES + cat_cols].columns.get_loc(col) for col in cat_cols]
    cat = CatBoostClassifier(
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
    cat_x_train = train_fit[NUMERIC_FEATURES + cat_cols].copy()
    cat_x_test = test[NUMERIC_FEATURES + cat_cols].copy()
    cat.fit(Pool(cat_x_train, y, cat_features=cat_features))
    preds["CatBoost path baseline"] = cat.predict_proba(Pool(cat_x_test, cat_features=cat_features))[:, 1]
    return preds


def rule_audit(test: pd.DataFrame, score: np.ndarray, out: Path) -> pd.DataFrame:
    audit = test.copy()
    audit["fepl_score"] = score
    k = max(1, int(math.ceil(0.10 * len(audit))))
    top = audit.sort_values("fepl_score", ascending=False).head(k)
    rules = [
        ("buffer_failure", "mu_buffer_failure_path"),
        ("late_recurrence", "mu_late_recurrence_path"),
        ("congested_carryover", "mu_congested_carryover_path"),
        ("upstream_delay", "mu_upstream_delay"),
        ("shortfall", "mu_recovery_shortfall"),
    ]
    rows = []
    total_pos = max(float(audit["severe_late_aircraft"].sum()), 1.0)
    for name, col in rules:
        selected = top[top[col].ge(0.50)]
        rows.append(
            {
                "rule": name,
                "top10_rows_with_membership_ge_0_5": int(len(selected)),
                "top10_positive_cases": int(selected["severe_late_aircraft"].sum()),
                "share_of_all_positive_cases": float(selected["severe_late_aircraft"].sum() / total_pos),
                "mean_membership_in_top10": float(top[col].mean()),
            }
        )
    rule_df = pd.DataFrame(rows).sort_values("share_of_all_positive_cases", ascending=False)
    rule_df.to_csv(out / "fepl_rule_audit.csv", index=False)
    examples = top[
        [
            "FlightDate",
            "carrier",
            "tail",
            "prev_origin",
            "origin",
            "dest",
            "prev_arr_delay",
            "scheduled_turnaround",
            "recovery_shortfall",
            "LateAircraftDelay",
            "fepl_score",
            "mu_buffer_failure_path",
            "mu_late_recurrence_path",
            "mu_congested_carryover_path",
        ]
    ].head(40)
    examples.to_csv(out / "fepl_top_path_examples.csv", index=False)
    return rule_df


def write_report(out: Path, metadata: dict, metrics: pd.DataFrame, weights: dict[str, float], rules: pd.DataFrame) -> None:
    lines = [
        "# FEPL Smoke Test",
        "",
        f"Train month: {metadata['train_month']}; test month: {metadata['test_month']}.",
        f"Carriers: {', '.join(metadata['carriers'])}.",
        f"Rotation paths: train {metadata['train_paths']:,}, test {metadata['test_paths']:,}.",
        f"Test severe late-aircraft positives: {metadata['test_positives']:,}.",
        "",
        "Higher Top-10% capture, area under the receiver operating characteristic curve (AUC), and average precision (AP) are better. Lower Brier score is better.",
        "",
        "| Model | Top-10% capture | Top-10% lift | AUC | AP | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['top10_capture']:.3f} | {row['top10_lift']:.3f} | "
            f"{row['auc']:.3f} | {row['pr_auc']:.3f} | {row['brier']:.3f} |"
        )
    lines += [
        "",
        "## FEPL weights",
        "",
        "| Path membership | Weight |",
        "|---|---:|",
    ]
    for name, value in weights.items():
        lines.append(f"| {name} | {value:.2f} |")
    lines += [
        "",
        "## Rule audit",
        "",
        "| Rule | Top-10% rows with membership >= 0.5 | Positive cases | Share of all positives | Mean membership |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in rules.iterrows():
        lines.append(
            f"| {row['rule']} | {row['top10_rows_with_membership_ge_0_5']} | "
            f"{row['top10_positive_cases']} | {row['share_of_all_positive_cases']:.3f} | "
            f"{row['mean_membership_in_top10']:.3f} |"
        )
    (out / "fepl_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--train-month", type=int, default=6)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--test-month", type=int, default=7)
    parser.add_argument("--carriers", default="WN,DL,AA,OO,UA")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--max-train-rows", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-panel", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    carriers = {item.strip().upper() for item in args.carriers.split(",") if item.strip()}
    panel_path = out / "fepl_rotation_path_panel.csv"
    if args.reuse_panel and panel_path.exists():
        paths = pd.read_csv(panel_path, parse_dates=["dep_dt", "arr_dt", "prev_arr_dt"])
    else:
        paths = build_rotation_paths(
            [(args.train_year, args.train_month), (args.test_year, args.test_month)],
            carriers,
            out,
        )
    train_ym = f"{args.train_year}_{args.train_month:02d}"
    test_ym = f"{args.test_year}_{args.test_month:02d}"
    train = paths[paths["ym"].eq(train_ym)].copy()
    test = paths[paths["ym"].eq(test_ym)].copy()
    if train.empty or test.empty:
        raise RuntimeError("Empty FEPL train or test split.")

    fepl_score, fepl_prob, weights = fit_fepl(train, test)
    y_test = test["severe_late_aircraft"].to_numpy(int)
    rows = [
        {"model": "FEPL fuzzy path lattice", **metric_dict(y_test, fepl_score, fepl_prob)},
        {"model": "Previous arrival delay", **metric_dict(y_test, test["prev_arr_delay"].to_numpy(float), None)},
        {"model": "Recovery shortfall", **metric_dict(y_test, test["recovery_shortfall"].to_numpy(float), None)},
        {"model": "Raw FEPL max membership", **metric_dict(y_test, test["mu_fepl_path"].to_numpy(float), test["mu_fepl_path"].to_numpy(float))},
    ]
    tree_preds = fit_tree_models(train, test, args.seed, args.max_train_rows)
    for model, pred in tree_preds.items():
        rows.append({"model": model, **metric_dict(y_test, pred, pred)})
    metrics = pd.DataFrame(rows).sort_values(["top10_capture", "auc"], ascending=False)
    metrics.to_csv(out / "fepl_smoke_metrics.csv", index=False)
    score_frame = test[["FlightDate", "carrier", "tail", "prev_origin", "origin", "dest", "LateAircraftDelay", "severe_late_aircraft"]].copy()
    score_frame["FEPL fuzzy path lattice"] = fepl_prob
    for model, pred in tree_preds.items():
        score_frame[model] = pred
    score_frame.to_csv(out / "fepl_smoke_predictions_sample.csv", index=False)
    rules = rule_audit(test, fepl_score, out)
    metadata = {
        "train_month": train_ym,
        "test_month": test_ym,
        "carriers": sorted(carriers),
        "train_paths": int(len(train)),
        "test_paths": int(len(test)),
        "test_positives": int(y_test.sum()),
        "temporal_validity": 1.0,
        "panel_rows": int(len(paths)),
        "positive_definition": "LateAircraftDelay >= 30 minutes",
    }
    (out / "fepl_smoke_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out, metadata, metrics, weights, rules)


if __name__ == "__main__":
    main()

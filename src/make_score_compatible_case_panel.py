from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kbs_frontier_smoke_tests import read_split
from kbs_sds_kb_strengthening import active_concepts
from sds_kb_evidence_quality import fit_full_sds_kb
from sds_kb_score_ceiling_audit import fit_lightgbm_score, top_mask


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "score_compatible_case_panel"


def compact_concepts(text: str) -> str:
    mapping = {
        "buffer failure": "Buffer",
        "late recurrence": "Recurrence",
        "congested carryover": "Congested",
        "upstream delay": "Upstream",
        "recovery shortfall": "Shortfall",
        "tight turnaround": "Tight turn",
    }
    parts = [mapping.get(part.strip(), part.strip()) for part in text.split(";") if part.strip()]
    return "; ".join(parts)


def path_string(row: pd.Series) -> str:
    return f"{row.get('prev_origin')}--{row.get('origin')}--{row.get('dest')}"


def add_case(
    rows: list[dict],
    seen: set[int],
    tag: str,
    idx: int,
    test_kb: pd.DataFrame,
    nearest_severe: pd.DataFrame,
    nearest_normal: pd.DataFrame,
) -> None:
    if idx in seen:
        return
    seen.add(idx)
    path = test_kb.iloc[idx]
    sev = nearest_severe.iloc[idx]
    norm = nearest_normal.iloc[idx]
    rows.append(
        {
            "type": tag,
            "carrier": path.get("carrier"),
            "path": path_string(path),
            "delay": float(path.get("LateAircraftDelay", np.nan)),
            "label": int(path.get("severe_late_aircraft", 0)),
            "risk": float(path.get("pe_kb_prob", np.nan)),
            "severe_distance": float(path.get("kb_dist_severe", np.nan)),
            "normal_distance": float(path.get("kb_dist_normal", np.nan)),
            "margin": float(path.get("kb_distance_margin", np.nan)),
            "active_concepts": compact_concepts(active_concepts(path)),
            "nearest_severe": f"{path_string(sev)}, {float(sev.get('LateAircraftDelay', np.nan)):.0f} min",
            "nearest_normal": f"{path_string(norm)}, {float(norm.get('LateAircraftDelay', np.nan)):.0f} min",
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = "2025_07_to_2025_08"
    train, test, _ = read_split(split)
    train_kb, test_kb, info, pe_score, pe_prob = fit_full_sds_kb(train, test, seed=2026, max_rows=120000)
    if info is None:
        raise RuntimeError("Nearest-prototype information is required.")
    lgbm_score = fit_lightgbm_score(train, test_kb, seed=2026, max_train_rows=120000)
    test_kb = test_kb.copy()
    test_kb["pe_kb_score"] = pe_score
    test_kb["pe_kb_prob"] = pe_prob
    test_kb["lgbm_score"] = lgbm_score
    nearest_severe = train.iloc[info["test_nearest_severe_train_idx"]].reset_index(drop=True)
    nearest_normal = train.iloc[info["test_nearest_normal_train_idx"]].reset_index(drop=True)

    y = test_kb["severe_late_aircraft"].to_numpy(int)
    pe_order = np.argsort(-pe_score)
    lgbm_order = np.argsort(-lgbm_score)
    pe_top = top_mask(pe_score, 0.10)
    lgbm_top = top_mask(lgbm_score, 0.10)

    rows: list[dict] = []
    seen: set[int] = set()
    for idx in pe_order:
        if y[idx] == 1 and test_kb.iloc[idx]["kb_distance_margin"] > 0:
            add_case(rows, seen, "PE-KB severe", int(idx), test_kb, nearest_severe, nearest_normal)
        if len(rows) >= 3:
            break
    for idx in lgbm_order:
        if y[idx] == 1 and lgbm_top[idx] and test_kb.iloc[idx]["kb_distance_margin"] > 0:
            add_case(rows, seen, "LightGBM severe", int(idx), test_kb, nearest_severe, nearest_normal)
        if len(rows) >= 6:
            break
    normal_candidates = [
        int(idx)
        for idx in pe_order
        if y[idx] == 0 and (pe_top[idx] or lgbm_top[idx])
    ]
    normal_candidates = sorted(
        normal_candidates,
        key=lambda i: abs(float(test_kb.iloc[i]["kb_distance_margin"])),
    )
    for idx in normal_candidates:
        add_case(rows, seen, "Boundary normal", idx, test_kb, nearest_severe, nearest_normal)
        if len(rows) >= 8:
            break

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "score_compatible_case_panel.csv", index=False)
    print((OUT / "score_compatible_case_panel.csv").resolve())


if __name__ == "__main__":
    main()

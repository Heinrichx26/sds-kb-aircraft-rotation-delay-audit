# SDS-KB Aircraft-Rotation Delay Audit

This repository contains the reproducibility package for the aircraft-rotation path-event knowledge base used in the manuscript on late-aircraft delay propagation audit.

## Contents

- `src/`: Python scripts for path construction, rolling validation, score-compatible audit checks, transfer validation, evidence-side ablation, and figure generation.
- `results/`: derived summary tables and reports used to check the manuscript results.
- `figures/`: exported data figures generated from the derived result summaries.
- `data/`: public data-source instructions. Raw flight records are excluded.

The repository excludes raw and semi-raw flight-record tables, manuscript files, submission documents, rendered paper PDFs, and local working artifacts.

## Public Data

The flight records are from the United States Bureau of Transportation Statistics Airline On-Time Performance database:

https://www.transtats.bts.gov/ONTIME/

The same public dataset is also indexed by Data.gov:

https://catalog.data.gov/dataset/u-s-marketing-air-carriers-on-time-performance

Download the required monthly airline on-time files and place them under:

```text
data/raw_open/bts_on_time/
```

The scripts build aircraft-rotation path panels and derived audit summaries from those monthly records.

## Environment

Create a Python environment and install the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Reproduction Outline

1. Download the public BTS monthly on-time records.
2. Build the rolling aircraft-rotation path panels:

```bash
python src/full_fepl_validation.py
```

3. Run the SDS-KB validation and audit-evidence checks:

```bash
python src/kbs_sds_kb_strengthening.py
python src/sds_kb_evidence_quality.py
python src/sds_kb_score_ceiling_audit.py
python src/sds_kb_evidence_side_ablation.py
python src/transfer_score_audit_baselines.py --protocol airport
python src/transfer_score_audit_baselines.py --protocol carrier
python src/transfer_score_audit_baselines.py --protocol cross_year
```

4. Regenerate derived tables and figures:

```bash
python src/sds_kb_risk_resolution_tables.py
python src/make_sds_kb_audit_figures.py
python src/make_kbs_r5_figures.py
```

The included `results/` files provide the derived summaries used for manuscript checking without redistributing the source flight records.


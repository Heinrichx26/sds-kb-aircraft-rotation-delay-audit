# Source Scripts

The scripts are organized around the paper workflow:

- `full_fepl_validation.py`: builds rolling aircraft-rotation path panels from public BTS records.
- `kbs_sds_kb_strengthening.py`: runs the main SDS-KB validation checks.
- `sds_kb_evidence_quality.py`: computes audit-evidence quality summaries.
- `sds_kb_score_ceiling_audit.py`: evaluates score-compatible evidence on a tree-score queue.
- `sds_kb_evidence_side_ablation.py`: evaluates evidence-side component roles.
- `transfer_score_audit_baselines.py`: runs airport, carrier, and cross-year transfer checks.
- `sds_kb_risk_resolution_tables.py`: assembles compact derived tables.
- `make_sds_kb_audit_figures.py` and `make_kbs_r5_figures.py`: regenerate data figures from derived result summaries.


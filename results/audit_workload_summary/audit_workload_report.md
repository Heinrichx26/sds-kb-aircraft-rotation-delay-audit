# Audit Workload Summary

Top-10% review workload over the 11 rolling splits.

| Model | Reviewed paths | Captured severe cases | Severe cases per 100 reviews | PR-AUC | Audit role |
|---|---:|---:|---:|---:|---|
| LightGBM path baseline | 407775 | 200200 | 49.10 | 0.9359 | score reference |
| Explainable Boosting Machine | 407775 | 196789 | 48.26 | 0.8629 | transparent score reference |
| FEPL fuzzy path rule model | 407775 | 191083 | 46.86 | 0.7714 | fixed fuzzy rule audit |
| Recovery shortfall | 407775 | 189547 | 46.48 | 0.7519 | single mechanism reference |
| Wang-Mendel fuzzy rule classifier | 407775 | 191475 | 46.96 | 0.8276 | large fuzzy rule reference |

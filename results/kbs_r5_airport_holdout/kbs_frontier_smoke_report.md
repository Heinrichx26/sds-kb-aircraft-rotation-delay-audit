# KBS Frontier Smoke Tests

Smoke tests use Jan-to-Feb and Jul-to-Aug 2025 splits from the cached aircraft-rotation path panels.
Higher Top-10% capture, AUC, and PR-AUC are better. Lower Brier score is better.

## Route Summary

| route                        |   splits |    rows |   positives |   mean_top10_capture |   mean_auc |   mean_pr_auc |   mean_brier |   mean_concept_count |   mean_feature_count | passes_pr_088   | passes_pr_085_t10_0965   | passes_action_t10_097   |
|:-----------------------------|---------:|--------:|------------:|---------------------:|-----------:|--------------:|-------------:|---------------------:|---------------------:|:----------------|:-------------------------|:------------------------|
| Recovery shortfall reference |       11 | 4077709 |      202145 |             0.942671 |   0.968763 |      0.751918 |    0.0813817 |                    1 |                    1 | False           | False                    | False                   |
| Current FEPL reference       |       11 | 4077709 |      202145 |             0.930742 |   0.968526 |      0.708428 |    0.024426  |                    2 |                    3 | False           | False                    | False                   |

## Selected Concepts



## Split Detail

| route                        | split              |   concept_count |   feature_count | selected_concepts         |   rows |   positives |   top10_capture |   top10_hit_rate |      auc |   pr_auc |     brier |
|:-----------------------------|:-------------------|----------------:|----------------:|:--------------------------|-------:|------------:|----------------:|-----------------:|---------:|---------:|----------:|
| Current FEPL reference       | 2025_01_to_2025_02 |               2 |               3 | existing FEPL memberships | 313896 |       12510 |        0.952198 |         0.379484 | 0.970749 | 0.669784 | 0.0221245 |
| Recovery shortfall reference | 2025_01_to_2025_02 |               1 |               1 | recovery shortfall        | 313896 |       12510 |        0.954596 |         0.38044  | 0.968887 | 0.708128 | 0.0724476 |
| Current FEPL reference       | 2025_02_to_2025_03 |               2 |               3 | existing FEPL memberships | 378077 |       15495 |        0.947596 |         0.388357 | 0.967424 | 0.679912 | 0.021827  |
| Recovery shortfall reference | 2025_02_to_2025_03 |               1 |               1 | recovery shortfall        | 378077 |       15495 |        0.949468 |         0.389124 | 0.967103 | 0.726149 | 0.0727259 |
| Current FEPL reference       | 2025_03_to_2025_04 |               2 |               3 | existing FEPL memberships | 372996 |       15580 |        0.950963 |         0.397212 | 0.969929 | 0.711211 | 0.0211215 |
| Recovery shortfall reference | 2025_03_to_2025_04 |               1 |               1 | recovery shortfall        | 372996 |       15580 |        0.953466 |         0.398257 | 0.969695 | 0.76163  | 0.0725316 |
| Current FEPL reference       | 2025_04_to_2025_05 |               2 |               3 | existing FEPL memberships | 388315 |       19642 |        0.929182 |         0.469999 | 0.96758  | 0.709487 | 0.0253127 |
| Recovery shortfall reference | 2025_04_to_2025_05 |               1 |               1 | recovery shortfall        | 388315 |       19642 |        0.947307 |         0.479167 | 0.96955  | 0.775153 | 0.0798115 |
| Current FEPL reference       | 2025_05_to_2025_06 |               2 |               3 | existing FEPL memberships | 382303 |       26670 |        0.88009  |         0.613952 | 0.964451 | 0.732555 | 0.0317033 |
| Recovery shortfall reference | 2025_05_to_2025_06 |               1 |               1 | recovery shortfall        | 382303 |       26670 |        0.911736 |         0.636028 | 0.965427 | 0.768089 | 0.0971716 |
| Current FEPL reference       | 2025_06_to_2025_07 |               2 |               3 | existing FEPL memberships | 388190 |       27986 |        0.871186 |         0.628069 | 0.959292 | 0.720294 | 0.0338    |
| Recovery shortfall reference | 2025_06_to_2025_07 |               1 |               1 | recovery shortfall        | 388190 |       27986 |        0.902594 |         0.650712 | 0.961648 | 0.766417 | 0.097509  |
| Current FEPL reference       | 2025_07_to_2025_08 |               2 |               3 | existing FEPL memberships | 375518 |       18515 |        0.925682 |         0.456407 | 0.964689 | 0.70354  | 0.024778  |
| Recovery shortfall reference | 2025_07_to_2025_08 |               1 |               1 | recovery shortfall        | 375518 |       18515 |        0.942371 |         0.464636 | 0.965266 | 0.753329 | 0.0800894 |
| Current FEPL reference       | 2025_08_to_2025_09 |               2 |               3 | existing FEPL memberships | 363977 |       11393 |        0.965066 |         0.302077 | 0.975886 | 0.726872 | 0.0157618 |
| Recovery shortfall reference | 2025_08_to_2025_09 |               1 |               1 | recovery shortfall        | 363977 |       11393 |        0.963311 |         0.301528 | 0.975478 | 0.780331 | 0.0643969 |
| Current FEPL reference       | 2025_09_to_2025_10 |               2 |               3 | existing FEPL memberships | 396028 |       16316 |        0.958384 |         0.394844 | 0.974815 | 0.731549 | 0.0197873 |
| Recovery shortfall reference | 2025_09_to_2025_10 |               1 |               1 | recovery shortfall        | 396028 |       16316 |        0.958997 |         0.395096 | 0.974998 | 0.782991 | 0.0831658 |
| Current FEPL reference       | 2025_10_to_2025_11 |               2 |               3 | existing FEPL memberships | 356333 |       16607 |        0.955802 |         0.445445 | 0.974281 | 0.738139 | 0.0212736 |
| Recovery shortfall reference | 2025_10_to_2025_11 |               1 |               1 | recovery shortfall        | 356333 |       16607 |        0.958511 |         0.446708 | 0.97355  | 0.759013 | 0.074509  |
| Current FEPL reference       | 2025_11_to_2025_12 |               2 |               3 | existing FEPL memberships | 362076 |       21431 |        0.902011 |         0.533888 | 0.964687 | 0.669361 | 0.0311958 |
| Recovery shortfall reference | 2025_11_to_2025_12 |               1 |               1 | recovery shortfall        | 362076 |       21431 |        0.927022 |         0.548691 | 0.964789 | 0.689865 | 0.100841  |

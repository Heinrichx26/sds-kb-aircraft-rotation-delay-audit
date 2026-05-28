# SDS-KB risk-resolution tables

These tables support the revised positioning: score ranking is reported separately from knowledge-base audit evidence.

## Rolling score references

| model              |   splits |    rows |   positives |   top10_capture |    auc |   pr_auc |   brier |
|:-------------------|---------:|--------:|------------:|----------------:|-------:|---------:|--------:|
| LightGBM           |       11 | 4077709 |      202145 |          0.9924 | 0.9963 |   0.9359 |  0.0211 |
| XGBoost            |       11 | 4077709 |      202145 |          0.9879 | 0.9956 |   0.9260 |  0.0242 |
| CatBoost           |       11 | 4077709 |      202145 |          0.9868 | 0.9954 |   0.9197 |  0.0249 |
| SDS-KB             |       11 | 4077709 |      202145 |          0.9866 | 0.9950 |   0.9132 |  0.0131 |
| Wang-Mendel        |       11 | 4077709 |      202145 |          0.9506 | 0.9737 |   0.8276 |  0.0187 |
| EBM                |       11 | 4077709 |      202145 |          0.9766 | 0.9920 |   0.8629 |  0.0159 |
| FEPL               |       11 | 4077709 |      202145 |          0.9497 | 0.9716 |   0.8092 |  0.0161 |
| Recovery shortfall |       11 | 4077709 |      202145 |          0.9427 | 0.9688 |   0.7519 |  0.0814 |

## Paired split-level tests

| comparison                   | metric        | better   |   mean_advantage |   wins |   ties |   splits |   wilcoxon_p_greater |
|:-----------------------------|:--------------|:---------|-----------------:|-------:|-------:|---------:|---------------------:|
| SDS-KB vs LightGBM           | top10_capture | higher   |        -0.005773 |      0 |      0 |       11 |             1.000000 |
| SDS-KB vs LightGBM           | pr_auc        | higher   |        -0.022671 |      0 |      0 |       11 |             1.000000 |
| SDS-KB vs LightGBM           | brier         | lower    |         0.008044 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs XGBoost            | top10_capture | higher   |        -0.001305 |      1 |      0 |       11 |             0.997559 |
| SDS-KB vs XGBoost            | pr_auc        | higher   |        -0.012810 |      0 |      0 |       11 |             1.000000 |
| SDS-KB vs XGBoost            | brier         | lower    |         0.011181 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs CatBoost           | top10_capture | higher   |        -0.000229 |      3 |      0 |       11 |             0.681152 |
| SDS-KB vs CatBoost           | pr_auc        | higher   |        -0.006496 |      0 |      0 |       11 |             1.000000 |
| SDS-KB vs CatBoost           | brier         | lower    |         0.011879 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs EBM                | top10_capture | higher   |         0.009985 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs EBM                | pr_auc        | higher   |         0.050284 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs EBM                | brier         | lower    |         0.002881 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Wang-Mendel        | top10_capture | higher   |         0.036007 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Wang-Mendel        | pr_auc        | higher   |         0.085581 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Wang-Mendel        | brier         | lower    |         0.005631 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs FEPL               | top10_capture | higher   |         0.036955 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs FEPL               | pr_auc        | higher   |         0.103987 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs FEPL               | brier         | lower    |         0.003070 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Recovery shortfall | top10_capture | higher   |         0.043943 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Recovery shortfall | pr_auc        | higher   |         0.161264 |     11 |      0 |       11 |             0.000488 |
| SDS-KB vs Recovery shortfall | brier         | lower    |         0.068321 |     11 |      0 |       11 |             0.000488 |

## Audit utility

| model   |   reviewed_paths_top10 |   captured_severe_cases |   evidence_supported_severe_cases |   severe_cases_per_100_reviews |   evidence_supported_cases_per_100_reviews |   mean_evidence_hit_rate |
|:--------|-----------------------:|------------------------:|----------------------------------:|-------------------------------:|-------------------------------------------:|-------------------------:|
| SDS-KB  |                 407775 |                  198930 |                            194119 |                        48.7843 |                                    47.6044 |                   0.9745 |

## Component role diagnostics

| variant                   |   splits |    rows |   positives |   top10_capture |    auc |   pr_auc |   brier |
|:--------------------------|---------:|--------:|------------:|----------------:|-------:|---------:|--------:|
| Full SDS-KB               |       11 | 4077709 |      202145 |          0.9866 | 0.9950 |   0.9132 |  0.0131 |
| No simulated severe store |       11 | 4077709 |      202145 |          0.9873 | 0.9952 |   0.9143 |  0.0128 |
| No prototype affinity     |       11 | 4077709 |      202145 |          0.9887 | 0.9955 |   0.9181 |  0.0129 |
| No fuzzy concepts         |       11 | 4077709 |      202145 |          0.9864 | 0.9950 |   0.9132 |  0.0129 |
| Prototype-only scorer     |       11 | 4077709 |      202145 |          0.9641 | 0.9864 |   0.8150 |  0.0197 |
| No monotone calibration   |       11 | 4077709 |      202145 |          0.9866 | 0.9950 |   0.9132 |  0.0183 |

## Cross-year transfer

| split              | variant                           |   use_prototypes |   use_fuzzy |   use_simulation |   use_tree |   use_calibration |   neighbors |   max_proto |   simulation_multiplier |   feature_count |    rows |   positives |   top10_capture |   top10_hit_rate |    auc |   pr_auc |   brier |
|:-------------------|:----------------------------------|-----------------:|------------:|-----------------:|-----------:|------------------:|------------:|------------:|------------------------:|----------------:|--------:|------------:|----------------:|-----------------:|-------:|---------:|--------:|
| train2024_test2025 | SDS-KB 2024-to-2025               |           1.0000 |      1.0000 |           1.0000 |     1.0000 |            1.0000 |      5.0000 |   4500.0000 |                  1.0000 |         20.0000 | 4401659 |      213611 |          0.9870 |           0.4790 | 0.9941 |   0.8945 |  0.0213 |
| train2024_test2025 | Recovery 2024-to-2025             |         nan      |    nan      |         nan      |   nan      |          nan      |    nan      |    nan      |                nan      |        nan      | 4401659 |      213611 |          0.9472 |           0.4596 | 0.9683 |   0.7458 |  0.0805 |
| train2024_test2025 | Raw fuzzy membership 2024-to-2025 |         nan      |    nan      |         nan      |   nan      |          nan      |    nan      |    nan      |                nan      |        nan      | 4401659 |      213611 |          0.9375 |           0.4550 | 0.9683 |   0.7029 |  0.0243 |

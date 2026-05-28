# Transfer score-compatible audit baselines

Each protocol first forms a score queue. PE-KB prototype evidence is then attached to the queued paths through the fixed path-event evidence channel.

## Summary

| protocol   | queue                           |   folds |    rows |   positives |   reviewed_paths |   captured_severe_cases |   evidence_supported_severe_cases |   pooled_evidence_hit_rate |   pooled_severe_cases_per_100_reviews |   pooled_evidence_supported_cases_per_100_reviews |   top10_capture_mean |   top10_capture_std |   auc_mean |   auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   brier_std |   evidence_hit_rate_mean |   evidence_hit_rate_std |
|:-----------|:--------------------------------|--------:|--------:|------------:|-----------------:|------------------------:|----------------------------------:|---------------------------:|--------------------------------------:|--------------------------------------------------:|---------------------:|--------------------:|-----------:|----------:|--------------:|-------------:|-------------:|------------:|-------------------------:|------------------------:|
| cross_year | PE-KB queue                     |       1 | 4401659 |      213611 |           440166 |                  210832 |                            204445 |                     0.9697 |                               47.8983 |                                           46.4472 |               0.9870 |              0.0000 |     0.9941 |    0.0000 |        0.8945 |       0.0000 |       0.0213 |      0.0000 |                   0.9697 |                  0.0000 |
| cross_year | LightGBM score + PE-KB evidence |       1 | 4401659 |      213611 |           440166 |                  212378 |                            204360 |                     0.9622 |                               48.2495 |                                           46.4279 |               0.9942 |              0.0000 |     0.9962 |    0.0000 |        0.9351 |       0.0000 |       0.0183 |      0.0000 |                   0.9622 |                  0.0000 |

## Fold detail

| protocol   | fold               | queue                           |    rows |   positives |   top10_capture |    auc |   pr_auc |   brier |   reviewed_paths |   captured_severe_cases |   evidence_supported_severe_cases |   evidence_hit_rate |   severe_cases_per_100_reviews |   evidence_supported_cases_per_100_reviews |
|:-----------|:-------------------|:--------------------------------|--------:|------------:|----------------:|-------:|---------:|--------:|-----------------:|------------------------:|----------------------------------:|--------------------:|-------------------------------:|-------------------------------------------:|
| cross_year | train2024_test2025 | PE-KB queue                     | 4401659 |      213611 |          0.9870 | 0.9941 |   0.8945 |  0.0213 |           440166 |                  210832 |                            204445 |              0.9697 |                        47.8983 |                                    46.4472 |
| cross_year | train2024_test2025 | LightGBM score + PE-KB evidence | 4401659 |      213611 |          0.9942 | 0.9962 |   0.9351 |  0.0183 |           440166 |                  212378 |                            204360 |              0.9622 |                        48.2495 |                                    46.4279 |

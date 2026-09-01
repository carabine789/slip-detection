# Curated Experiment Logs

This directory keeps only the logs needed to support the current report and
public handoff. Older exploratory logs were removed from the curated set.

## Final Model

- `20260827_final_train_augdiag59_refdiff_instance_augtranslate.log`  
  Final training run for the augdiag59 model.

- `20260827_final_eval_comprehensive_plusdiag_threshold050.log`  
  Final model evaluation on the comprehensive `final_plus_diag` evaluation set
  at threshold 0.5.

- `20260829_threshold_scan_comprehensive_plusdiag.log`  
  Threshold sensitivity scan on the comprehensive `final_plus_diag` evaluation
  set.

- `20260829_threshold_scan_old_final_test.log`  
  Threshold sensitivity scan on the older final-test CSV.

- `20260831_final_eval_clean_diag062_065_threshold050_070.log`  
  Final model evaluation on the strictly non-overlapping diagonal subset
  `session_062` to `session_065` at thresholds 0.5 and 0.7.

## Cross-Validation

- `20260827_cv_augdiag59_fold00_train.log`
- `20260827_cv_augdiag59_fold01_train.log`
- `20260827_cv_augdiag59_fold02_train.log`
- `20260827_cv_augdiag59_fold03_train.log`
- `20260827_cv_augdiag59_fold04_train.log`

These five logs support the 5-fold session-level cross-validation summary for
the augdiag59 dataset.

## Baseline Comparisons

- `20260827_baseline_old_v2_eval_comprehensive_plusdiag.log`  
  Older V2 model evaluated on the comprehensive `final_plus_diag` set.

- `20260827_baseline_old_v2_eval_clean_diag062_065.log`  
  Older V2 model evaluated on the clean non-overlapping diagonal subset
  `session_062` to `session_065`.

## Diagnostics

- `session_013_health_same_group.json`  
  Session-level diagnostic artifact for the earlier `session_013` anomaly.

- `session_051_065_contact_sheet.jpg`  
  Contact sheet for the later added sessions.

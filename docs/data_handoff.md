# Data Handoff Notes

This repository does not track the raw RGB data or checkpoints. The local
handoff package is split into:

- `handoff_core/`: core data, split CSVs, session manifest, and final
  checkpoints corresponding to the current report.
- `handoff_archive/`: older split CSVs, earlier checkpoints, excluded sessions,
  and compressed backups kept only for reference.

## Recommended Entry Points

Use these files first:

| Purpose | File |
|---|---|
| Normalized session index | `handoff_core/session_manifest.csv` |
| Final train split | `handoff_core/data/raw_cnn/train_windows_multi_material_augdiag.csv` |
| Final validation split | `handoff_core/data/raw_cnn/val_windows_multi_material_augdiag.csv` |
| Final independent test split | `handoff_core/data/raw_cnn_test/test_windows_final_plus_diag.csv` |
| Final 5-fold CV splits | `handoff_core/data/raw_cnn/cv_multi_material_augdiag_59/` |
| Final best checkpoint | `handoff_core/checkpoints/model_v2_augdiag_59_augprob08_p05_warm3/slip_cnn_v2_best.pth` |

## Metadata Normalization

The original per-session `metadata.csv` files are preserved as collected, but
they have two historical schema variants:

- `v1`: uses `label_0_desc` and `label_1_desc` to describe the two labels.
- `v2`: keeps those fields and also adds `non_slip_behavior` and
  `slip_behavior`.

For handoff, use `handoff_core/session_manifest.csv` as the normalized index.
It maps both schema variants into one consistent row format:

| Field | Meaning |
|---|---|
| `split` | `train`, `val`, or `test` |
| `data_root` | Relative root containing the session folder |
| `session` | Local session folder name |
| `session_key` | Unique key combining split and session name |
| `material` | Contact material |
| `label_0_desc` | Original description for label 0 |
| `label_1_desc` | Original description for label 1 |
| `non_slip_behavior` | Normalized non-slip behavior |
| `slip_behavior` | Normalized slip behavior |
| `slip_motion` | Session-level slip motion type |
| `frames` | Number of captured frames |
| `actual_fps` | Measured capture FPS |
| `windows` | Number of windows used in the final split CSV |
| `label0_windows` | Number of label-0 windows |
| `label1_windows` | Number of label-1 windows |
| `reference` | Reference image filename |
| `metadata_schema` | Original metadata schema variant, `v1` or `v2` |
| `metadata_path` | Path to original session metadata |
| `frames_csv_path` | Path to original frame-level CSV |
| `window_csv_path` | Final window CSV that includes this session |

The train/validation and test roots are separate. For example,
`data/raw_cnn/session_001` and `data/raw_cnn_test/session_001` are different
sessions, so `session_key` should be used when a globally unique identifier is
needed.

## Archive Policy

Files under `handoff_archive/` are not the main report split. They are kept for
traceability only. The current report and README correspond to the files listed
in the recommended entry points above.

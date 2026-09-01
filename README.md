# Slip Detection CNN

This module implements a binary slip / non-slip classifier from short windows
of consecutive RGB tactile frames collected with the NLiPsTac sensor.

The current baseline uses an 8-frame window, a per-session reference frame, and
a compact CNN trained on session-level splits. The main implementation is
`slip_cnn_v2.py`.

## Layout

```text
slip_detection/
+-- slip_cnn.py              # Early baseline model
+-- slip_cnn_v2.py           # Current model: InstanceNorm, augmentation, replay support
+-- eval_on_test.py          # Read-only checkpoint evaluation on a CSV split
+-- data_collect/            # Capture, window generation, CV split, diagnostics
+-- docs/                    # Progress reports and diagnostic figures
+-- logs/                    # Curated logs for report writing
+-- logs_archive/            # Local full log history, ignored by Git
+-- checkpoints/             # Local model checkpoints, ignored by Git
`-- data/                    # Local raw frames and window CSVs, ignored by Git
```

Large raw data, checkpoints, and local archives are intentionally not tracked by
Git. Share them separately if needed.

## 1. Collect Data

Example collection command from the repository root:

```bash
python slip_detection/data_collect/raw_cnn_collect.py collect \
  --no-serial --preview --duration-sec 24 --target-fps 30 --min-frames 8 \
  --width 1920 --height 1080 --backend dshow --camera-fps 30 --fourcc MJPG \
  --default-label -1 --label-schedule 0:-1,5:0,15:-1,18:1 \
  --material plastic \
  --label-0-desc "non-slip pressing" \
  --label-1-desc "slip" \
  --slip-motion translation
```

`--label-schedule` maps elapsed time to labels:

- `-1`: ignored transition frames
- `0`: non-slip
- `1`: slip

Each session is saved under `slip_detection/data/raw_cnn/session_NNN/`.

## 2. Build Window CSVs

Use session-level splits to avoid window leakage between train and validation:

```bash
python slip_detection/data_collect/raw_cnn_collect.py windows \
  --data-root slip_detection/data/raw_cnn \
  --window 8 --stride 8 --val-stride 16 \
  --split-by-session \
  --val-sessions session_003,session_008,session_012,session_018 \
  --label-policy last --balance-labels \
  --train-csv slip_detection/data/raw_cnn/train_windows.csv \
  --val-csv slip_detection/data/raw_cnn/val_windows.csv
```

For k-fold session-level cross-validation:

```bash
python slip_detection/data_collect/session_cv_splits.py \
  --source-csv slip_detection/data/raw_cnn/train_windows.csv \
  --source-csv slip_detection/data/raw_cnn/val_windows.csv \
  --out-dir slip_detection/data/raw_cnn/cv_splits \
  --folds 5
```

## 3. Train

Current recommended baseline:

```bash
python slip_detection/slip_cnn_v2.py train \
  --data-root slip_detection/data/raw_cnn \
  --train-csv slip_detection/data/raw_cnn/train_windows.csv \
  --val-csv slip_detection/data/raw_cnn/val_windows.csv \
  --out-dir slip_detection/checkpoints/model_v2 \
  --width 320 --height 180 \
  --input-mode refdiff \
  --norm instance \
  --architecture stacked \
  --grad-clip 1.0 \
  --lr-patience 3 --lr-factor 0.5 \
  --smooth-window 3 --early-stop-patience 8 \
  --augment-translate --aug-translate-frac 0.08 \
  --augment-prob 0.5 --augment-warmup-epochs 3 \
  --epochs 40 --batch-size 32 --num-workers 4
```

The best checkpoint is selected using smoothed balanced accuracy instead of a
single-epoch peak.

## 4. Evaluate

`eval_on_test.py` performs read-only checkpoint evaluation:

```bash
python slip_detection/eval_on_test.py \
  --checkpoint slip_detection/checkpoints/model_v2/slip_cnn_v2_best.pth \
  --data-root slip_detection/data/raw_cnn_test \
  --csv slip_detection/data/raw_cnn_test/test_windows.csv \
  --width 320 --height 180 \
  --thresholds 0.5
```

Multiple thresholds can be scanned for sensitivity analysis:

```bash
python slip_detection/eval_on_test.py \
  --checkpoint slip_detection/checkpoints/model_v2/slip_cnn_v2_best.pth \
  --data-root slip_detection/data/raw_cnn_test \
  --csv slip_detection/data/raw_cnn_test/test_windows.csv \
  --width 320 --height 180 \
  --thresholds 0.45,0.5,0.55,0.6,0.65,0.7
```

Use threshold scans as sensitivity analysis unless the threshold was selected
on validation data before evaluating the test set.

## Input Modes

- `raw`: use RGB frames directly.
- `refdiff`: subtract the per-session reference frame from each current frame.
- `framediff`: use differences between consecutive frames.
- `ref-and-raw`: concatenate raw frames and reference-difference frames.

## Reporting Notes

The latest concise report is available at:

- [`docs/progress_report_20260829.md`](docs/progress_report_20260829.md)
- [`docs/progress_report_20260829.tex`](docs/progress_report_20260829.tex)

When reporting results, distinguish between:

- session-level cross-validation results;
- comprehensive regression/evaluation sets that may include historical
  sessions or windows;
- strictly non-overlapping new-session subsets.

This distinction is important for avoiding overstatement of held-out
generalization.

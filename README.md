# Slip Detection CNN

Binary slip / non-slip classification from short windows of consecutive RGB tactile frames, built on top of the NLiPsTac sensor used elsewhere in this repository. See [`docs/progress_report_20260718.md`](docs/progress_report_20260718.md) for the full methodology writeup (data collection design, the session_013 anomaly investigation, training stabilization, cross-validation results).

## Layout

```
slip_detection/
├── slip_cnn.py           # baseline model: stacked-frame input, BatchNorm/GroupNorm
├── slip_cnn_v2.py         # + InstanceNorm and a shared-encoder (per-frame + temporal aggregation) architecture
├── eval_on_test.py         # one-shot, read-only evaluation of a checkpoint on a held-out CSV
├── data_collect/           # camera capture, session-to-window CSV building, session-level CV splits
├── data/                   # raw_cnn/ (train+val sessions), raw_cnn_test/ (held-out test sessions) — gitignored, not in git
├── logs/                   # curated training logs supporting the progress report
├── logs_archive/           # full log history from every run — gitignored, local only
├── checkpoints/            # early single-split checkpoints — gitignored, not in git
└── docs/                   # progress report + session_013 diagnostic images
```

`data/`, `logs_archive/`, and `checkpoints/` are all gitignored — the images and model weights are too large for git. Re-collect data with the commands below, or share `data/` and checkpoints separately (e.g. compressed archive, cloud storage) if handing this off to someone else.

## 1. Collect data

Run from repo root:

```bash
python slip_detection/data_collect/raw_cnn_collect.py collect \
  --no-serial --preview --duration-sec 24 --target-fps 30 --min-frames 8 \
  --width 1920 --height 1080 --backend dshow --camera-fps 30 --fourcc MJPG \
  --default-label -1 --label-schedule 0:-1,5:0,15:-1,18:1 \
  --material plastic --label-0-desc "non-slip pressing" --label-1-desc "slip" \
  --session-note "plastic; 1920x1080 MJPG; label 0 non-slip; label 1 slip"
```

`--label-schedule` maps seconds to labels (`-1` = ignored/transition). Each session is saved under `slip_detection/data/raw_cnn/session_NNN/`.

## 2. Build window CSVs

Session-level split (no window leakage between train/val):

```bash
python slip_detection/data_collect/raw_cnn_collect.py windows \
  --data-root slip_detection/data/raw_cnn --window 8 --stride 8 --val-stride 16 \
  --split-by-session --val-sessions session_003,session_008,session_012,session_018 \
  --label-policy last --balance-labels \
  --train-csv slip_detection/data/raw_cnn/train_windows.csv \
  --val-csv slip_detection/data/raw_cnn/val_windows.csv
```

For k-fold cross-validation over all sessions, use `slip_detection/data_collect/session_cv_splits.py`.

## 3. Train

```bash
python slip_detection/slip_cnn.py train \
  --data-root slip_detection/data/raw_cnn \
  --train-csv slip_detection/data/raw_cnn/train_windows.csv \
  --val-csv slip_detection/data/raw_cnn/val_windows.csv \
  --out-dir slip_detection/checkpoints \
  --width 320 --height 180 --input-mode ref-and-raw --norm batch \
  --grad-clip 1.0 --lr-patience 3 --lr-factor 0.5 --smooth-window 3 --early-stop-patience 8 \
  --epochs 40 --batch-size 32 --num-workers 4
```

`slip_cnn_v2.py` takes the same arguments plus `--norm instance` and `--architecture {stacked,shared}` (with `--temporal-agg {mean,max,meanmax,gru}` for `shared`).

Best checkpoint is selected on a smoothed (moving-average) `balanced_acc` rather than a single-epoch spike, with automatic LR decay on plateau and early stopping.

## 4. Evaluate on a held-out test set

`eval_on_test.py` is read-only — it never writes to the checkpoint or training pipeline, so the test set stays untouched by any tuning loop:

```bash
python slip_detection/eval_on_test.py \
  --checkpoint slip_detection/checkpoints/slip_cnn_best.pth \
  --data-root slip_detection/data/raw_cnn_test \
  --csv slip_detection/data/raw_cnn_test/test_windows_final.csv \
  --width 320 --height 180
```

## Input modes

- `raw` — frames as-is
- `refdiff` — each frame minus a per-session reference (background) frame
- `framediff` — consecutive-frame differences
- `ref-and-raw` — raw frames concatenated with their ref-diff (best so far)

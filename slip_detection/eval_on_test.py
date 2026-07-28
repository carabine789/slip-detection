"""
One-shot evaluation of a trained checkpoint against a held-out CSV (e.g. the
final test set). Does not train, does not touch the checkpoint, does not
write anything back into the training pipeline -- pure read-only inference
+ metrics reporting.

Usage:
    python slip_detection/eval_on_test.py \
        --checkpoint slip_detection/checkpoints/slip_cnn_v2_best.pth \
        --data-root slip_detection/data/raw_cnn_test \
        --csv slip_detection/data/raw_cnn_test/test_windows_final.csv \
        --width 320 --height 180 --batch-size 32 --num-workers 4
"""

import argparse

import torch
from torch.utils.data import DataLoader

from slip_cnn_v2 import (
    DEVICE,
    RawFrameWindowDataset,
    SharedEncoderSlipCNN,
    MultiFrameSlipCNN,
    evaluate,
    print_group_metrics,
)


def build_model_from_checkpoint(state: dict, window_size: int):
    architecture = state.get("architecture", "stacked")
    norm_type = state.get("norm_type", "batch")
    temporal_agg = state.get("temporal_agg", "meanmax")
    if architecture == "shared":
        model = SharedEncoderSlipCNN(
            window_size=window_size,
            input_mode=state["input_mode"],
            norm_type=norm_type,
            temporal_agg=temporal_agg,
        ).to(DEVICE)
    else:
        model = MultiFrameSlipCNN(
            window_size=window_size,
            input_mode=state["input_mode"],
            norm_type=norm_type,
        ).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    return model, architecture


def main():
    parser = argparse.ArgumentParser(description="One-shot checkpoint evaluation on a held-out CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"[INFO] device={DEVICE}", flush=True)
    print(f"[INFO] loading checkpoint: {args.checkpoint}", flush=True)
    state = torch.load(args.checkpoint, map_location=DEVICE)

    window_size = state.get("window_size", 8)
    input_mode = state.get("input_mode", "refdiff")
    diff_threshold = state.get("diff_threshold", 0.03)
    architecture = state.get("architecture", "stacked")

    print(
        f"[INFO] checkpoint metadata: epoch={state.get('epoch')} "
        f"best_metric={state.get('best_metric_name')}={state.get('best_metric_value')} "
        f"input_mode={input_mode} norm={state.get('norm_type')} architecture={architecture} "
        f"temporal_agg={state.get('temporal_agg')}",
        flush=True,
    )

    model, architecture = build_model_from_checkpoint(state, window_size)

    print(f"[INFO] loading eval dataset: {args.csv}", flush=True)
    ds = RawFrameWindowDataset(
        args.data_root, args.csv, args.width, args.height, window_size,
        input_mode, diff_threshold, architecture=architecture,
    )
    print(f"[INFO] eval samples={len(ds)}", flush=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    metrics = evaluate(model, loader)
    print(
        f"[FINAL-TEST] acc={metrics['acc']:.4f} balanced_acc={metrics['balanced_acc']:.4f} "
        f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"specificity={metrics['specificity']:.4f} pred_pos={metrics['pred_pos_rate']:.4f} "
        f"tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}",
        flush=True,
    )
    print_group_metrics(metrics, max_groups=50)


if __name__ == "__main__":
    main()

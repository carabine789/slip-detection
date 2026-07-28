"""
CNN-based slip detection, v2 experimental variant of slip_cnn.py.

This file is a self-contained fork of slip_cnn.py that adds two
generalization-oriented options on top of the original "stacked" (early
channel-fusion) approach:

  1. InstanceNorm2d as an additional --norm choice (per-sample normalization,
     which cancels out global per-session appearance/lighting bias instead of
     relying on batch/group statistics).

  2. A "shared" architecture: a per-frame CNN encoder (shared weights across
     all frames in a window) followed by explicit temporal aggregation
     (mean / max / meanmax pooling, or a small GRU) instead of concatenating
     frames along the channel dimension. This gives the model an explicit
     inductive bias that a window is a *sequence* of frames, rather than one
     image with extra channels, which should make it harder for the model to
     shortcut on session-specific appearance and easier for it to learn
     genuine motion/slip features.

The original slip_cnn.py is left untouched; this file can be run the same
way but with two extra CLI flags: --architecture {stacked,shared} and
--temporal-agg {mean,max,meanmax,gru} (only used when --architecture=shared),
plus "instance" added to the --norm choices.
"""

import argparse
import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_raw_rgb(path: str, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def parse_frame_list(value: str) -> List[str]:
    frames = [item.strip() for item in value.split(";") if item.strip()]
    if not frames:
        raise ValueError("Empty frames field in CSV")
    return frames


def threshold_motion(diff: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return diff
    return np.where(diff >= threshold, diff, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# "stacked" (original) input pipeline: concatenate frames along channel axis
# ---------------------------------------------------------------------------

def build_input_frames(
    frames: List[np.ndarray],
    input_mode: str,
    reference: Optional[np.ndarray] = None,
    diff_threshold: float = 0.03,
) -> List[np.ndarray]:
    if input_mode == "raw":
        return frames
    if input_mode == "refdiff":
        if reference is None:
            raise ValueError("reference image is required for input_mode=refdiff")
        return [frame - reference for frame in frames]
    if input_mode == "framediff":
        if len(frames) < 2:
            raise ValueError("framediff requires at least 2 frames")
        diffs = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i - 1])
            diffs.append(threshold_motion(diff, diff_threshold))
        return diffs
    if input_mode == "ref-and-raw":
        if reference is None:
            raise ValueError("reference image is required for input_mode=ref-and-raw")
        return frames + [frame - reference for frame in frames]
    raise ValueError(f"Unknown input_mode: {input_mode}")


def input_channels_for(window_size: int, input_mode: str) -> int:
    if input_mode == "framediff":
        return 3 * (window_size - 1)
    multiplier = 2 if input_mode == "ref-and-raw" else 1
    return 3 * window_size * multiplier


def to_sequence_tensor(
    frames: List[np.ndarray],
    input_mode: str = "raw",
    reference: Optional[np.ndarray] = None,
    diff_threshold: float = 0.03,
) -> torch.Tensor:
    seq_frames = build_input_frames(frames, input_mode, reference, diff_threshold)
    seq = np.concatenate(seq_frames, axis=2)
    seq = np.transpose(seq, (2, 0, 1))
    return torch.from_numpy(seq).float()


# ---------------------------------------------------------------------------
# "shared" input pipeline: keep frames separate -> (T, C, H, W) sequence
# ---------------------------------------------------------------------------

def per_frame_channels_for(input_mode: str) -> int:
    # ref-and-raw here means "raw frame concatenated with its ref-diff" per
    # timestep (6 channels), not doubling the number of timesteps.
    if input_mode == "ref-and-raw":
        return 6
    return 3


def build_per_frame_stack(
    frames: List[np.ndarray],
    input_mode: str,
    reference: Optional[np.ndarray] = None,
    diff_threshold: float = 0.03,
) -> List[np.ndarray]:
    if input_mode == "raw":
        return frames
    if input_mode == "refdiff":
        if reference is None:
            raise ValueError("reference image is required for input_mode=refdiff")
        return [frame - reference for frame in frames]
    if input_mode == "framediff":
        if len(frames) < 2:
            raise ValueError("framediff requires at least 2 frames")
        diffs = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i - 1])
            diffs.append(threshold_motion(diff, diff_threshold))
        return diffs
    if input_mode == "ref-and-raw":
        if reference is None:
            raise ValueError("reference image is required for input_mode=ref-and-raw")
        return [np.concatenate([frame, frame - reference], axis=2) for frame in frames]
    raise ValueError(f"Unknown input_mode: {input_mode}")


def to_frame_stack_tensor(
    frames: List[np.ndarray],
    input_mode: str = "raw",
    reference: Optional[np.ndarray] = None,
    diff_threshold: float = 0.03,
) -> torch.Tensor:
    stack_frames = build_per_frame_stack(frames, input_mode, reference, diff_threshold)
    arr = np.stack([np.transpose(f, (2, 0, 1)) for f in stack_frames], axis=0)
    return torch.from_numpy(arr).float()


class RawFrameWindowDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        width: int = 320,
        height: int = 240,
        window_size: Optional[int] = None,
        input_mode: str = "raw",
        diff_threshold: float = 0.03,
        architecture: str = "stacked",
    ):
        self.root = root
        self.csv_path = csv_path
        self.size = (width, height)
        self.samples = []
        self.window_size = window_size
        self.input_mode = input_mode
        self.diff_threshold = diff_threshold
        self.architecture = architecture
        self._reference_cache: Dict[str, np.ndarray] = {}

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"frames", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for row in reader:
                frame_paths = parse_frame_list(row["frames"])
                if self.window_size is None:
                    self.window_size = len(frame_paths)
                if len(frame_paths) != self.window_size:
                    raise ValueError(
                        f"Window size mismatch in {csv_path}: expected {self.window_size}, got {len(frame_paths)}"
                    )
                self.samples.append({
                    "frames": frame_paths,
                    "reference": row.get("reference", ""),
                    "label": row["label"],
                    "session": row.get("session", ""),
                    "material": row.get("material", ""),
                    "behavior": row.get("behavior", ""),
                    "motion_type": row.get("motion_type", ""),
                    "non_slip_behavior": row.get("non_slip_behavior", ""),
                    "slip_behavior": row.get("slip_behavior", ""),
                    "slip_motion": row.get("slip_motion", ""),
                })

        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")
        if self.window_size is None or self.window_size < 2:
            raise ValueError("window_size must be at least 2")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        frame_arrays = []
        for path in row["frames"]:
            if not os.path.isabs(path):
                path = os.path.join(self.root, path)
            frame_arrays.append(read_raw_rgb(path, self.size))
        reference = None
        if self.input_mode in ("refdiff", "ref-and-raw"):
            ref_path = row.get("reference", "")
            if not ref_path:
                raise ValueError("CSV is missing reference column/path for reference-difference input")
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(self.root, ref_path)
            reference = self._reference_cache.get(ref_path)
            if reference is None:
                reference = read_raw_rgb(ref_path, self.size)
                self._reference_cache[ref_path] = reference
        if self.architecture == "shared":
            x = to_frame_stack_tensor(frame_arrays, self.input_mode, reference, self.diff_threshold)
        else:
            x = to_sequence_tensor(frame_arrays, self.input_mode, reference, self.diff_threshold)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = {
            "session": row.get("session", ""),
            "material": row.get("material", ""),
            "behavior": row.get("behavior", ""),
            "motion_type": row.get("motion_type", ""),
            "non_slip_behavior": row.get("non_slip_behavior", ""),
            "slip_behavior": row.get("slip_behavior", ""),
            "slip_motion": row.get("slip_motion", ""),
        }
        return x, y, meta


def make_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "group":
        if num_channels % 8 == 0:
            num_groups = 8
        elif num_channels % 4 == 0:
            num_groups = 4
        else:
            num_groups = 1
        return nn.GroupNorm(num_groups, num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    return nn.BatchNorm2d(num_channels)


class MultiFrameSlipCNN(nn.Module):
    """Original early-fusion (channel-stack) architecture, unchanged."""

    def __init__(self, window_size: int = 8, input_mode: str = "raw", norm_type: str = "batch"):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        self.input_mode = input_mode
        self.norm_type = norm_type
        in_channels = input_channels_for(window_size, input_mode)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            make_norm(norm_type, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat).squeeze(1)


class PerFrameEncoder(nn.Module):
    """Shared-weight CNN backbone applied independently to each frame."""

    def __init__(self, in_channels: int, norm_type: str = "batch"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            make_norm(norm_type, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            make_norm(norm_type, 192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class SharedEncoderSlipCNN(nn.Module):
    """Per-frame shared encoder + explicit temporal aggregation.

    Input shape: (B, T, C, H, W) where T is the number of timesteps in the
    window (T = window_size, or window_size - 1 for framediff).
    """

    def __init__(
        self,
        window_size: int = 8,
        input_mode: str = "raw",
        norm_type: str = "batch",
        temporal_agg: str = "meanmax",
    ):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        self.input_mode = input_mode
        self.norm_type = norm_type
        self.temporal_agg = temporal_agg

        in_channels = per_frame_channels_for(input_mode)
        self.encoder = PerFrameEncoder(in_channels, norm_type)
        feat_dim = 192

        self.gru = None
        if temporal_agg == "gru":
            gru_hidden = 128
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=gru_hidden, batch_first=True)
            classifier_in = gru_hidden
        elif temporal_agg == "meanmax":
            classifier_in = feat_dim * 2
        elif temporal_agg in ("mean", "max"):
            classifier_in = feat_dim
        else:
            raise ValueError(f"Unknown temporal_agg: {temporal_agg}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        feat = self.encoder(x.view(b * t, c, h, w))
        feat = feat.view(b, t, -1)
        if self.temporal_agg == "gru":
            _, hidden = self.gru(feat)
            pooled = hidden[-1]
        elif self.temporal_agg == "mean":
            pooled = feat.mean(dim=1)
        elif self.temporal_agg == "max":
            pooled, _ = feat.max(dim=1)
        else:  # meanmax
            mean_feat = feat.mean(dim=1)
            max_feat, _ = feat.max(dim=1)
            pooled = torch.cat([mean_feat, max_feat], dim=1)
        return self.classifier(pooled).squeeze(1)


def build_model(args, window_size: int) -> nn.Module:
    if args.architecture == "shared":
        return SharedEncoderSlipCNN(
            window_size=window_size,
            input_mode=args.input_mode,
            norm_type=args.norm,
            temporal_agg=args.temporal_agg,
        ).to(DEVICE)
    return MultiFrameSlipCNN(
        window_size=window_size,
        input_mode=args.input_mode,
        norm_type=args.norm,
    ).to(DEVICE)


@dataclass
class SlipResult:
    is_slipping: bool
    probability: float
    ready: bool


class RealtimeSlipDetector:
    def __init__(
        self,
        model_path: str,
        width: int = 320,
        height: int = 240,
        threshold: float = 0.5,
        window_size: int = 8,
        input_mode: str = "raw",
        reference_path: Optional[str] = None,
        diff_threshold: float = 0.03,
        norm_type: str = "batch",
        architecture: str = "stacked",
        temporal_agg: str = "meanmax",
    ):
        self.size = (width, height)
        self.threshold = threshold
        state = torch.load(model_path, map_location=DEVICE)
        if isinstance(state, dict) and "window_size" in state:
            window_size = int(state["window_size"])
        if isinstance(state, dict) and "input_mode" in state:
            input_mode = state["input_mode"]
        if isinstance(state, dict) and "diff_threshold" in state:
            diff_threshold = float(state["diff_threshold"])
        if isinstance(state, dict) and "norm_type" in state:
            norm_type = state["norm_type"]
        if isinstance(state, dict) and "architecture" in state:
            architecture = state["architecture"]
        if isinstance(state, dict) and "temporal_agg" in state:
            temporal_agg = state["temporal_agg"]
        self.window_size = window_size
        self.input_mode = input_mode
        self.diff_threshold = diff_threshold
        self.norm_type = norm_type
        self.architecture = architecture
        self.temporal_agg = temporal_agg
        self.frames = deque(maxlen=self.window_size)
        self.reference = read_raw_rgb(reference_path, self.size) if reference_path else None
        if self.architecture == "shared":
            self.model = SharedEncoderSlipCNN(
                window_size=self.window_size,
                input_mode=self.input_mode,
                norm_type=self.norm_type,
                temporal_agg=self.temporal_agg,
            ).to(DEVICE)
        else:
            self.model = MultiFrameSlipCNN(
                window_size=self.window_size,
                input_mode=self.input_mode,
                norm_type=self.norm_type,
            ).to(DEVICE)
        weights = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(weights)
        self.model.eval()

    @torch.inference_mode()
    def update(self, frame_bgr: np.ndarray) -> SlipResult:
        frame = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.frames.append(rgb)
        if len(self.frames) < self.window_size:
            return SlipResult(False, 0.0, False)

        if self.architecture == "shared":
            x = to_frame_stack_tensor(list(self.frames), self.input_mode, self.reference, self.diff_threshold).unsqueeze(0).to(DEVICE)
        else:
            x = to_sequence_tensor(list(self.frames), self.input_mode, self.reference, self.diff_threshold).unsqueeze(0).to(DEVICE)
        logit = self.model(x)
        prob = torch.sigmoid(logit)[0].item()
        return SlipResult(prob >= self.threshold, prob, True)


def train_one_epoch(model, loader, optimizer, log_every: int = 20, grad_clip: float = 0.0):
    model.train()
    total = 0.0
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        x, y = batch[:2]
        optimizer.zero_grad(set_to_none=True)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader)):
            avg_loss = total / batch_idx
            elapsed = max(1e-9, time.time() - t0)
            print(
                f"  [train] batch {batch_idx}/{len(loader)} "
                f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                f"speed={batch_idx / elapsed:.2f} batch/s",
                flush=True,
            )
    return total / max(1, len(loader))


def metric_summary(stats: Dict[str, float]) -> Dict[str, float]:
    tp = int(stats.get("tp", 0))
    tn = int(stats.get("tn", 0))
    fp = int(stats.get("fp", 0))
    fn = int(stats.get("fn", 0))
    count = max(1, tp + tn + fp + fn)
    prob_sum = float(stats.get("prob_sum", 0.0))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "acc": (tp + tn) / count,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": 0.5 * (recall + specificity),
        "f1": f1,
        "pred_pos_rate": (tp + fp) / count,
        "mean_prob": prob_sum / count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "count": count,
    }


def update_metric_counts(stats: Dict[str, float], pred_value: bool, y_value: bool, prob_value: float) -> None:
    if pred_value and y_value:
        stats["tp"] = stats.get("tp", 0) + 1
    elif (not pred_value) and (not y_value):
        stats["tn"] = stats.get("tn", 0) + 1
    elif pred_value and (not y_value):
        stats["fp"] = stats.get("fp", 0) + 1
    else:
        stats["fn"] = stats.get("fn", 0) + 1
    stats["prob_sum"] = stats.get("prob_sum", 0.0) + float(prob_value)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_stats: Dict[str, float] = {}
    prob_by_class = {"pos": [], "neg": []}
    groups = {"behavior": {}, "motion_type": {}, "session": {}, "material": {}}
    for batch in loader:
        x, y = batch[:2]
        meta = batch[2] if len(batch) > 2 else None
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        prob = torch.sigmoid(model(x))
        pred = prob >= 0.5
        y_bool = y >= 0.5
        batch_size = y.numel()
        for i in range(batch_size):
            pred_i = bool(pred[i].item())
            y_i = bool(y_bool[i].item())
            prob_i = float(prob[i].item())
            update_metric_counts(total_stats, pred_i, y_i, prob_i)
            prob_by_class["pos" if y_i else "neg"].append(prob_i)
            if isinstance(meta, dict):
                for group_name in groups:
                    values = meta.get(group_name, "")
                    if isinstance(values, (list, tuple)):
                        value = str(values[i]) if i < len(values) else ""
                    else:
                        value = str(values)
                    if not value:
                        value = "unknown"
                    group_stats = groups[group_name].setdefault(value, {})
                    update_metric_counts(group_stats, pred_i, y_i, prob_i)
    metrics = metric_summary(total_stats)
    metrics["groups"] = {name: {key: metric_summary(stats) for key, stats in values.items()} for name, values in groups.items()}
    metrics["mean_prob_pos"] = sum(prob_by_class["pos"]) / max(1, len(prob_by_class["pos"]))
    metrics["mean_prob_neg"] = sum(prob_by_class["neg"]) / max(1, len(prob_by_class["neg"]))
    return metrics


def print_group_metrics(metrics: dict, max_groups: int = 12) -> None:
    groups = metrics.get("groups", {})
    for group_name in ("behavior", "motion_type", "session", "material"):
        group_values = groups.get(group_name, {})
        if not group_values:
            continue
        ranked = sorted(
            group_values.items(),
            key=lambda item: (item[1].get("fp", 0) + item[1].get("fn", 0), item[1].get("count", 0)),
            reverse=True,
        )
        print(f"  [val/{group_name}]", flush=True)
        for value, stats in ranked[:max_groups]:
            errors = stats.get("fp", 0) + stats.get("fn", 0)
            print(
                f"    {value}: n={stats.get('count', 0)} acc={stats.get('acc', 0.0):.4f} "
                f"err={errors} tp={stats.get('tp', 0)} tn={stats.get('tn', 0)} "
                f"fp={stats.get('fp', 0)} fn={stats.get('fn', 0)} "
                f"pred_pos={stats.get('pred_pos_rate', 0.0):.4f}",
                flush=True,
            )


def make_checkpoint(
    model,
    args,
    epoch: int,
    window_size: int,
    train_loss: float,
    metrics: Optional[dict],
    checkpoint_kind: str,
    best_metric_name: str,
    best_metric_value: float,
) -> dict:
    return {
        "model": model.state_dict(),
        "checkpoint_kind": checkpoint_kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "metrics": metrics,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "window_size": window_size,
        "input_mode": args.input_mode,
        "diff_threshold": args.diff_threshold,
        "norm_type": args.norm,
        "architecture": args.architecture,
        "temporal_agg": args.temporal_agg,
        "width": args.width,
        "height": args.height,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
    }


def train(args):
    print(f"[INFO] device={DEVICE}", flush=True)
    print(f"[INFO] loading train dataset: {args.train_csv}", flush=True)
    train_ds = RawFrameWindowDataset(
        args.data_root, args.train_csv, args.width, args.height, args.window_size,
        args.input_mode, args.diff_threshold, architecture=args.architecture,
    )
    print(
        f"[INFO] train samples={len(train_ds)} window={train_ds.window_size} "
        f"input_mode={args.input_mode} norm={args.norm} architecture={args.architecture} "
        f"temporal_agg={args.temporal_agg} size={args.width}x{args.height}",
        flush=True,
    )
    print(f"[INFO] loading val dataset: {args.val_csv}", flush=True)
    val_ds = (
        RawFrameWindowDataset(
            args.data_root, args.val_csv, args.width, args.height, train_ds.window_size,
            args.input_mode, args.diff_threshold, architecture=args.architecture,
        )
        if args.val_csv else None
    )
    if val_ds is not None:
        print(f"[INFO] val samples={len(val_ds)}", flush=True)
    window_size = train_ds.window_size

    model = build_model(args, window_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    persistent = args.num_workers > 0
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=persistent,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs) if val_ds else None
    train_eval_loader = (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if args.eval_train else None
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_metric_name = "smoothed_balanced_acc"
    best_metric_value = -1.0
    last_loss = 0.0
    last_metrics = None
    recent_balanced_acc: deque = deque(maxlen=max(1, args.smooth_window))
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"[INFO] epoch {epoch}/{args.epochs} started", flush=True)
        loss = train_one_epoch(model, train_loader, optimizer, args.log_every, args.grad_clip)
        last_loss = loss
        msg = f"[Epoch {epoch:03d}] loss={loss:.4f} window={window_size}"
        if train_eval_loader is not None:
            train_metrics = evaluate(model, train_eval_loader)
            msg += (
                f" train_acc={train_metrics['acc']:.4f}"
                f" train_balanced_acc={train_metrics['balanced_acc']:.4f}"
            )
        if val_loader is not None:
            metrics = evaluate(model, val_loader)
            last_metrics = metrics
            acc = metrics["acc"]
            recent_balanced_acc.append(metrics["balanced_acc"])
            smoothed_balanced_acc = sum(recent_balanced_acc) / len(recent_balanced_acc)
            metrics["smoothed_balanced_acc"] = smoothed_balanced_acc
            current_metric = metrics[best_metric_name]
            scheduler.step(smoothed_balanced_acc)
            current_lr = optimizer.param_groups[0]["lr"]
            msg += (
                f" val_acc={acc:.4f}"
                f" balanced_acc={metrics['balanced_acc']:.4f}"
                f" smoothed_balanced_acc={smoothed_balanced_acc:.4f}"
                f" f1={metrics['f1']:.4f}"
                f" precision={metrics['precision']:.4f}"
                f" recall={metrics['recall']:.4f}"
                f" pred_pos={metrics['pred_pos_rate']:.4f}"
                f" mean_p={metrics['mean_prob']:.4f}"
                f" mean_p_pos={metrics['mean_prob_pos']:.4f}"
                f" mean_p_neg={metrics['mean_prob_neg']:.4f}"
                f" tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}"
                f" lr={current_lr:.2e}"
            )
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                epochs_without_improvement = 0
                torch.save(
                    make_checkpoint(model, args, epoch, window_size, loss, metrics, "best", best_metric_name, best_metric_value),
                    os.path.join(args.out_dir, "slip_cnn_v2_best.pth"),
                )
                print(f"[BEST] epoch={epoch} {best_metric_name}={best_metric_value:.4f} saved", flush=True)
            else:
                epochs_without_improvement += 1

        msg += f" time={time.time() - t0:.1f}s"
        print(msg)
        if val_loader is not None:
            print_group_metrics(metrics)

        if val_loader is not None and args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"[EARLY-STOP] no {best_metric_name} improvement for {epochs_without_improvement} epochs, "
                f"stopping at epoch {epoch}",
                flush=True,
            )
            break

    torch.save(
        make_checkpoint(model, args, epoch, window_size, last_loss, last_metrics, "last", best_metric_name, best_metric_value),
        os.path.join(args.out_dir, "slip_cnn_v2_last.pth"),
    )


def realtime(args):
    detector = RealtimeSlipDetector(
        args.model,
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        window_size=args.window_size,
        input_mode=args.input_mode,
        reference_path=args.reference,
        diff_threshold=args.diff_threshold,
        norm_type=args.norm,
        architecture=args.architecture,
        temporal_agg=args.temporal_agg,
    )
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = detector.update(frame)
            if result.ready:
                color = (0, 0, 255) if result.is_slipping else (0, 180, 0)
                text = f"slip={int(result.is_slipping)} p={result.probability:.2f} window={detector.window_size}"
            else:
                color = (120, 120, 120)
                text = f"warming {len(detector.frames)}/{detector.window_size}"
            cv2.putText(frame, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.imshow("multi-frame raw slip detection (v2)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Slip detection v2: shared-encoder architecture + InstanceNorm option")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--data-root", default="slip_detection/data/raw_cnn")
    train_p.add_argument("--train-csv", default="slip_detection/data/raw_cnn/train_windows.csv")
    train_p.add_argument("--val-csv", default="slip_detection/data/raw_cnn/val_windows.csv")
    train_p.add_argument("--out-dir", default="slip_detection/checkpoints")
    train_p.add_argument("--width", type=int, default=320)
    train_p.add_argument("--height", type=int, default=240)
    train_p.add_argument("--window-size", type=int, default=None, help="Infer from CSV when omitted")
    train_p.add_argument("--input-mode", choices=["raw", "refdiff", "framediff", "ref-and-raw"], default="framediff")
    train_p.add_argument("--norm", choices=["batch", "group", "instance"], default="batch", help="CNN normalization layer")
    train_p.add_argument("--architecture", choices=["stacked", "shared"], default="stacked",
                          help="stacked = original channel-concat fusion; shared = per-frame shared encoder + temporal aggregation")
    train_p.add_argument("--temporal-agg", choices=["mean", "max", "meanmax", "gru"], default="meanmax",
                          help="Only used when --architecture=shared")
    train_p.add_argument("--diff-threshold", type=float, default=0.03, help="Threshold for abs frame difference in [0,1]")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--num-workers", type=int, default=0, help="0 loads images on the main process (slow, serial disk+decode). Set to your CPU core count (e.g. 4-8) to parallelize image reading across worker processes.")
    train_p.add_argument("--prefetch-factor", type=int, default=2, help="Batches each worker prefetches ahead; only used when --num-workers > 0")
    train_p.add_argument("--log-every", type=int, default=20, help="Print training progress every N batches; 0 disables batch logs")
    train_p.add_argument("--eval-train", action="store_true", help="Also evaluate on the training set each epoch to check train/val gap")
    train_p.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm; 0 disables clipping")
    train_p.add_argument("--lr-patience", type=int, default=3, help="Epochs with no smoothed-metric improvement before LR is reduced")
    train_p.add_argument("--lr-factor", type=float, default=0.5, help="Factor to multiply LR by on plateau")
    train_p.add_argument("--min-lr", type=float, default=1e-6, help="Floor for LR decay")
    train_p.add_argument("--smooth-window", type=int, default=3, help="Number of recent epochs averaged for checkpoint selection/LR scheduling")
    train_p.add_argument("--early-stop-patience", type=int, default=8, help="Stop after this many epochs with no smoothed-metric improvement; 0 disables early stopping")

    rt_p = sub.add_parser("realtime")
    rt_p.add_argument("--model", required=True)
    rt_p.add_argument("--cam-index", type=int, default=0)
    rt_p.add_argument("--capture-width", type=int, default=1280)
    rt_p.add_argument("--capture-height", type=int, default=720)
    rt_p.add_argument("--width", type=int, default=320)
    rt_p.add_argument("--height", type=int, default=240)
    rt_p.add_argument("--window-size", type=int, default=8, help="Overridden by checkpoint metadata when available")
    rt_p.add_argument("--input-mode", choices=["raw", "refdiff", "framediff", "ref-and-raw"], default="framediff", help="Overridden by checkpoint metadata when available")
    rt_p.add_argument("--norm", choices=["batch", "group", "instance"], default="batch", help="Overridden by checkpoint metadata when available")
    rt_p.add_argument("--architecture", choices=["stacked", "shared"], default="stacked", help="Overridden by checkpoint metadata when available")
    rt_p.add_argument("--temporal-agg", choices=["mean", "max", "meanmax", "gru"], default="meanmax", help="Overridden by checkpoint metadata when available")
    rt_p.add_argument("--diff-threshold", type=float, default=0.03, help="Threshold for abs frame difference in [0,1]")
    rt_p.add_argument("--reference", default=None, help="Reference image for refdiff/ref-and-raw realtime inference")
    rt_p.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.cmd == "train":
        train(cli_args)
    elif cli_args.cmd == "realtime":
        realtime(cli_args)

"""Check whether a raw CNN session is an image-statistics outlier.

This script is intentionally independent from model training. It compares one
target session against peer sessions using reference/frame brightness, color,
sharpness, reference difference, label timing, and metadata consistency.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_metadata(session_dir: str) -> Dict[str, str]:
    path = os.path.join(session_dir, "metadata.csv")
    metadata = {"session": os.path.basename(session_dir)}
    if not os.path.isfile(path):
        return metadata
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                metadata[row[0].strip()] = row[1].strip()
    return metadata


def read_frame_rows(session_dir: str) -> List[dict]:
    path = os.path.join(session_dir, "frames.csv")
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_image(path: str, width: int) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if width > 0 and image.shape[1] > width:
        scale = width / float(image.shape[1])
        size = (width, max(1, int(round(image.shape[0] * scale))))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def image_stats(image: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)
    return {
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "mean_b": float(b.mean()),
        "mean_g": float(g.mean()),
        "mean_r": float(r.mean()),
        "mean_h": float(hsv[:, :, 0].mean()),
        "mean_s": float(hsv[:, :, 1].mean()),
        "mean_v": float(hsv[:, :, 2].mean()),
        "sharp_lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def hist_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def select_rows(rows: Sequence[dict], label: Optional[int], max_frames: int) -> List[dict]:
    filtered = []
    for row in rows:
        try:
            row_label = int(float(row.get("label", "-999")))
        except ValueError:
            continue
        if label is None:
            if row_label < 0:
                continue
        elif row_label != label:
            continue
        filtered.append(row)
    if max_frames <= 0 or len(filtered) <= max_frames:
        return filtered
    idx = np.linspace(0, len(filtered) - 1, max_frames).round().astype(int)
    return [filtered[int(i)] for i in idx]


def frame_path(session_dir: str, row: dict) -> str:
    rel = row.get("frame", "").replace("\\", "/")
    return os.path.join(session_dir, rel)


def aggregate_dicts(values: Sequence[Dict[str, float]], prefix: str) -> Dict[str, float]:
    if not values:
        return {}
    keys = sorted(values[0])
    out = {}
    for key in keys:
        arr = np.array([v[key] for v in values if key in v], dtype=np.float64)
        if arr.size == 0:
            continue
        out[f"{prefix}_{key}_mean"] = float(arr.mean())
        out[f"{prefix}_{key}_std"] = float(arr.std())
    return out


def safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def behavior_key(metadata: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        metadata.get("non_slip_behavior", ""),
        metadata.get("slip_behavior", ""),
        metadata.get("slip_motion", ""),
    )


def list_sessions(data_root: str) -> List[str]:
    sessions = []
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "rgb")):
            sessions.append(name)
    return sorted(sessions)


def analyze_session(data_root: str, session: str, max_frames: int, resize_width: int) -> Dict[str, object]:
    session_dir = os.path.join(data_root, session)
    metadata = read_metadata(session_dir)
    rows = read_frame_rows(session_dir)
    reference_name = metadata.get("reference", "reference.jpg")
    reference_path = os.path.join(session_dir, reference_name)
    reference = read_image(reference_path, resize_width)

    metrics: Dict[str, float] = {
        "frames_csv_rows": float(len(rows)),
        "actual_fps": safe_float(metadata.get("actual_fps", "")),
        "target_fps": safe_float(metadata.get("target_fps", "")),
        "reported_frames": safe_float(metadata.get("frames", "")),
        "reference_file_size": float(os.path.getsize(reference_path)) if os.path.isfile(reference_path) else float("nan"),
    }
    if reference is not None:
        metrics.update({f"reference_{k}": v for k, v in image_stats(reference).items()})
    else:
        metrics["reference_missing"] = 1.0

    label_counts: Dict[str, int] = defaultdict(int)
    label_times: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        label = row.get("label", "")
        label_counts[label] += 1
        label_times[label].append(safe_float(row.get("timestamp_sec", "")))
    for label, count in label_counts.items():
        metrics[f"label_{label}_frames"] = float(count)
        times = [t for t in label_times[label] if not math.isnan(t)]
        if times:
            metrics[f"label_{label}_start_sec"] = float(min(times))
            metrics[f"label_{label}_end_sec"] = float(max(times))
            metrics[f"label_{label}_duration_sec"] = float(max(times) - min(times))

    for label in (0, 1):
        selected = select_rows(rows, label, max_frames)
        stats = []
        diff_stats = []
        hist_distances = []
        for row in selected:
            image = read_image(frame_path(session_dir, row), resize_width)
            if image is None:
                continue
            stats.append(image_stats(image))
            if reference is not None:
                if reference.shape[:2] != image.shape[:2]:
                    ref = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    ref = reference
                diff = cv2.absdiff(image, ref)
                diff_stats.append(image_stats(diff))
                hist_distances.append(float(cv2.compareHist(hist_gray(image), hist_gray(ref), cv2.HISTCMP_BHATTACHARYYA)))
        metrics.update(aggregate_dicts(stats, f"label{label}"))
        metrics.update(aggregate_dicts(diff_stats, f"label{label}_refdiff"))
        if hist_distances:
            metrics[f"label{label}_hist_ref_bhatt_mean"] = float(np.mean(hist_distances))
            metrics[f"label{label}_hist_ref_bhatt_std"] = float(np.std(hist_distances))

    return {
        "session": session,
        "metadata": metadata,
        "behavior_key": behavior_key(metadata),
        "metrics": metrics,
    }


def robust_z(value: float, baseline: Sequence[float]) -> float:
    arr = np.array([v for v in baseline if not math.isnan(v)], dtype=np.float64)
    if arr.size < 3 or math.isnan(value):
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-9:
        std = float(arr.std())
        return (value - med) / std if std > 1e-9 else 0.0
    return 0.6745 * (value - med) / mad


def compare_target(target: dict, baselines: Sequence[dict], z_threshold: float) -> List[dict]:
    keys = sorted(target["metrics"])
    rows = []
    for key in keys:
        value = target["metrics"][key]
        if not isinstance(value, (int, float)) or math.isnan(float(value)):
            continue
        baseline_values = [b["metrics"].get(key, float("nan")) for b in baselines]
        z = robust_z(float(value), baseline_values)
        if math.isnan(z):
            continue
        baseline_clean = [float(v) for v in baseline_values if isinstance(v, (int, float)) and not math.isnan(float(v))]
        if not baseline_clean:
            continue
        rows.append({
            "metric": key,
            "target": float(value),
            "baseline_median": float(np.median(baseline_clean)),
            "baseline_min": float(np.min(baseline_clean)),
            "baseline_max": float(np.max(baseline_clean)),
            "robust_z": float(z),
            "flag": abs(z) >= z_threshold,
        })
    rows.sort(key=lambda row: abs(row["robust_z"]), reverse=True)
    return rows


def print_report(target: dict, baselines: Sequence[dict], comparisons: Sequence[dict], top_k: int) -> None:
    meta = target["metadata"]
    print(f"[target] {target['session']}")
    print(
        "[behavior] "
        f"non_slip={meta.get('non_slip_behavior', '')} | "
        f"slip={meta.get('slip_behavior', '')} | "
        f"motion={meta.get('slip_motion', '')}"
    )
    print(f"[baseline_sessions] {', '.join(b['session'] for b in baselines)}")
    print("\n[metadata]")
    for key in [
        "created_at",
        "material",
        "camera_index",
        "requested_width",
        "requested_height",
        "actual_width",
        "actual_height",
        "light_cmd",
        "target_fps",
        "actual_fps",
        "frames",
        "reference",
    ]:
        print(f"  {key}: {meta.get(key, '')}")
    print("\n[top_outliers]")
    for row in comparisons[:top_k]:
        marker = "!" if row["flag"] else " "
        print(
            f"{marker} {row['metric']}: target={row['target']:.4f} "
            f"baseline_median={row['baseline_median']:.4f} "
            f"range=[{row['baseline_min']:.4f}, {row['baseline_max']:.4f}] "
            f"robust_z={row['robust_z']:.2f}"
        )
    flagged = [row for row in comparisons if row["flag"]]
    print(f"\n[summary] flagged_metrics={len(flagged)} / compared_metrics={len(comparisons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a target raw_cnn session with peer sessions.")
    parser.add_argument("--data-root", default="data/raw_cnn")
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--baseline-sessions", default="", help="Comma-separated baseline sessions. Default: all peers.")
    parser.add_argument("--same-behavior-group", action="store_true", help="Use only sessions with matching behavior metadata.")
    parser.add_argument("--max-frames-per-label", type=int, default=80)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    all_sessions = list_sessions(args.data_root)
    if args.target_session not in all_sessions:
        raise RuntimeError(f"Target session not found: {args.target_session}")

    target = analyze_session(args.data_root, args.target_session, args.max_frames_per_label, args.resize_width)
    if args.baseline_sessions:
        baseline_names = [item.strip() for item in args.baseline_sessions.split(",") if item.strip()]
    else:
        baseline_names = [session for session in all_sessions if session != args.target_session]
    if args.same_behavior_group:
        target_key = target["behavior_key"]
        baseline_names = [
            session for session in baseline_names
            if behavior_key(read_metadata(os.path.join(args.data_root, session))) == target_key
        ]
    if not baseline_names:
        raise RuntimeError("No baseline sessions selected")

    baselines = [
        analyze_session(args.data_root, session, args.max_frames_per_label, args.resize_width)
        for session in baseline_names
    ]
    comparisons = compare_target(target, baselines, args.z_threshold)
    print_report(target, baselines, comparisons, args.top_k)

    if args.json_out:
        payload = {"target": target, "baselines": baselines, "comparisons": comparisons}
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()

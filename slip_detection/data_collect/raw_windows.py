"""Sliding-window CSV generation for raw CNN slip datasets."""

import csv
import os
import random
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
WINDOW_FIELDNAMES = ["frames", "reference", "label", "session", "material", "behavior", "motion_type", "non_slip_behavior", "slip_behavior", "slip_motion"]


def normalize_rel(path: str) -> str:
    return path.replace("\\", "/")


def find_session_reference(data_root: str, session_name: str) -> Optional[str]:
    for name in ("reference.jpg", "reference.png", "reference.jpeg", "reference.bmp"):
        rel_path = normalize_rel(os.path.join(session_name, name))
        if os.path.isfile(os.path.join(data_root, rel_path)):
            return rel_path
    return None


def read_frame_labels(session_dir: str) -> Dict[str, int]:
    labels = {}
    frames_csv = os.path.join(session_dir, "frames.csv")
    if not os.path.isfile(frames_csv):
        return labels
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = normalize_rel(row.get("frame", ""))
            if frame:
                labels[frame] = int(float(row.get("label", -1) or -1))
    return labels


def parse_session_names(data_root: str, sessions_arg: str) -> List[str]:
    if sessions_arg:
        return [item.strip() for item in sessions_arg.split(",") if item.strip()]
    names = []
    for name in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, name, "rgb")):
            names.append(name)
    names.sort()
    return names



def read_session_metadata(session_dir: str) -> Dict[str, str]:
    metadata = {
        "session": os.path.basename(session_dir),
        "material": "",
        "label_0_desc": "",
        "label_1_desc": "",
        "non_slip_behavior": "",
        "slip_behavior": "",
        "slip_motion": "",
        "session_note": "",
    }
    metadata_csv = os.path.join(session_dir, "metadata.csv")
    if not os.path.isfile(metadata_csv):
        return metadata
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if key in metadata:
                metadata[key] = value
    return metadata


def window_metadata(session_meta: Dict[str, str], label: int) -> Dict[str, str]:
    # Older sessions record non_slip_behavior/slip_behavior explicitly; newer sessions
    # only record label_0_desc/label_1_desc, so fall back to those when the specific
    # fields are missing (label 0 is always non-slip, label 1 is always slip).
    non_slip_behavior = session_meta.get("non_slip_behavior", "") or session_meta.get("label_0_desc", "")
    slip_behavior = session_meta.get("slip_behavior", "") or session_meta.get("label_1_desc", "")
    behavior = slip_behavior if label == 1 else non_slip_behavior
    motion_type = session_meta.get("slip_motion", "") if label == 1 else "non-slip"
    return {
        "session": session_meta.get("session", ""),
        "material": session_meta.get("material", ""),
        "behavior": behavior,
        "motion_type": motion_type,
        "non_slip_behavior": non_slip_behavior,
        "slip_behavior": slip_behavior,
        "slip_motion": session_meta.get("slip_motion", ""),
    }

def write_rows_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: List[dict], val_ratio: float, seed: int):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = int(round(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def split_sessions(session_names: List[str], val_sessions_arg: str, val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if val_sessions_arg:
        val_set = {item.strip() for item in val_sessions_arg.split(",") if item.strip()}
        missing = sorted(val_set - set(session_names))
        if missing:
            raise RuntimeError(f"val sessions not found: {missing}")
        train_names = [name for name in session_names if name not in val_set]
        val_names = [name for name in session_names if name in val_set]
        return train_names, val_names

    names = list(session_names)
    random.Random(seed).shuffle(names)
    n_val = max(1, int(round(len(names) * val_ratio))) if val_ratio > 0 else 0
    val_names = sorted(names[:n_val])
    train_names = sorted(names[n_val:])
    return train_names, val_names


def label_counts(rows: List[dict]) -> Dict[str, int]:
    counts = {"0": 0, "1": 0}
    for row in rows:
        label = str(int(float(row["label"])))
        if label in counts:
            counts[label] += 1
    return counts


def balance_rows(rows: List[dict], seed: int) -> List[dict]:
    by_label = {"0": [], "1": []}
    for row in rows:
        label = str(int(float(row["label"])))
        if label not in by_label:
            continue
        by_label[label].append(row)
    target = min(len(by_label["0"]), len(by_label["1"]))
    if target == 0:
        raise RuntimeError(f"Cannot balance labels: counts={label_counts(rows)}")
    rng = random.Random(seed)
    balanced = rng.sample(by_label["0"], target) + rng.sample(by_label["1"], target)
    rng.shuffle(balanced)
    return balanced


def window_label(label_values: List[int], policy: str) -> int:
    if policy == "last":
        return label_values[-1]
    if policy == "majority":
        return int(sum(label_values) >= (len(label_values) / 2.0))
    return int(any(label_values))


def collect_windows(data_root: str, session_names: List[str], window: int, stride: int, label_policy: str) -> List[dict]:
    rows = []
    total_skipped_missing = 0
    total_skipped_ignored = 0
    for session_name in session_names:
        session_dir = os.path.join(data_root, session_name)
        rgb_dir = os.path.join(session_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            print(f"[WARN] skip missing rgb dir: {rgb_dir}")
            continue

        reference = find_session_reference(data_root, session_name)
        if reference is None:
            print(f"[WARN] skip {session_name}: missing per-session reference image")
            continue

        labels = read_frame_labels(session_dir)
        session_meta = read_session_metadata(session_dir)
        images = sorted(name for name in os.listdir(rgb_dir) if os.path.splitext(name)[1].lower() in IMAGE_EXTS)
        if len(images) < window:
            print(f"[WARN] skip {session_name}: {len(images)} frames < window {window}")
            continue

        session_rows = 0
        skipped_missing = 0
        skipped_ignored = 0
        for start in range(0, len(images) - window + 1, stride):
            chunk = images[start:start + window]
            rel_frames = [normalize_rel(os.path.join(session_name, "rgb", name)) for name in chunk]
            local_frames = [normalize_rel(os.path.join("rgb", name)) for name in chunk]
            if any(frame not in labels for frame in local_frames):
                skipped_missing += 1
                continue
            y_values = [labels[frame] for frame in local_frames]
            if any(label < 0 for label in y_values):
                skipped_ignored += 1
                continue
            y = window_label(y_values, label_policy)
            row = {"frames": ";".join(rel_frames), "reference": reference, "label": y}
            row.update(window_metadata(session_meta, y))
            rows.append(row)
            session_rows += 1

        total_skipped_missing += skipped_missing
        total_skipped_ignored += skipped_ignored
        print(
            f"[INFO] {session_name}: windows={session_rows} "
            f"skipped_missing_label={skipped_missing} skipped_ignored={skipped_ignored}"
        )

    if total_skipped_missing or total_skipped_ignored:
        print(f"[INFO] skipped total: missing_label={total_skipped_missing}, ignored_label={total_skipped_ignored}")
    return rows


def make_windows(args) -> None:
    session_names = parse_session_names(args.data_root, args.sessions)
    if not session_names:
        raise RuntimeError(f"No sessions found under {args.data_root}")

    train_stride = args.train_stride if args.train_stride is not None else args.stride
    val_stride = args.val_stride if args.val_stride is not None else args.stride

    if args.split_by_session or args.val_sessions:
        train_sessions, val_sessions = split_sessions(session_names, args.val_sessions, args.val_ratio, args.seed)
        if not train_sessions:
            raise RuntimeError("No train sessions after session split")
        if not val_sessions:
            raise RuntimeError("No val sessions after session split")
        print(f"[INFO] train sessions: {','.join(train_sessions)}")
        print(f"[INFO] val sessions: {','.join(val_sessions)}")
        train_rows = collect_windows(args.data_root, train_sessions, args.window, train_stride, args.label_policy)
        val_rows = collect_windows(args.data_root, val_sessions, args.window, val_stride, args.label_policy)
        if not train_rows or not val_rows:
            raise RuntimeError(f"No windows generated: train={len(train_rows)} val={len(val_rows)}")
        print(f"[INFO] train label counts before balance: {label_counts(train_rows)}")
        print(f"[INFO] val label counts before balance: {label_counts(val_rows)}")
        if args.balance_labels:
            train_rows = balance_rows(train_rows, args.seed)
            print(f"[INFO] train label counts after balance: {label_counts(train_rows)}")
        if args.balance_val_labels:
            val_rows = balance_rows(val_rows, args.seed)
            print(f"[INFO] val label counts after balance: {label_counts(val_rows)}")
        write_rows_csv(args.train_csv, train_rows, WINDOW_FIELDNAMES)
        write_rows_csv(args.val_csv, val_rows, WINDOW_FIELDNAMES)
        print(f"[DONE] train={len(train_rows)} -> {args.train_csv}")
        print(f"[DONE] val={len(val_rows)} -> {args.val_csv}")
        return

    rows = collect_windows(args.data_root, session_names, args.window, args.stride, args.label_policy)
    if not rows:
        raise RuntimeError("No windows generated")

    raw_counts = label_counts(rows)
    print(f"[INFO] label counts before balance: {raw_counts}")
    if args.balance_labels:
        rows = balance_rows(rows, args.seed)
        print(f"[INFO] label counts after balance: {label_counts(rows)}")

    if args.val_ratio > 0:
        train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
        write_rows_csv(args.train_csv, train_rows, WINDOW_FIELDNAMES)
        write_rows_csv(args.val_csv, val_rows, WINDOW_FIELDNAMES)
        print(f"[DONE] train={len(train_rows)} -> {args.train_csv}")
        print(f"[DONE] val={len(val_rows)} -> {args.val_csv}")
    else:
        write_rows_csv(args.train_csv, rows, WINDOW_FIELDNAMES)
        print(f"[DONE] windows={len(rows)} -> {args.train_csv}")
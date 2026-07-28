"""Generate session-level cross-validation CSV splits for raw CNN windows."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def read_rows(paths: Sequence[str]) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    fieldnames: List[str] = []
    seen = set()
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise RuntimeError(f"CSV has no header: {path}")
            if not fieldnames:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise RuntimeError(f"CSV header mismatch: {path}")
            for row in reader:
                key = (row.get("session", ""), row.get("frames", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    if "session" not in fieldnames:
        raise RuntimeError("CSV must contain a 'session' column")
    return rows, fieldnames


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def session_key(rows: Iterable[dict], group_columns: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    grouped: Dict[str, Tuple[str, ...]] = {}
    for row in rows:
        session = row["session"]
        values = tuple(row.get(column, "") for column in group_columns)
        if session in grouped and grouped[session] != values:
            raise RuntimeError(
                f"Session {session} has inconsistent group values: "
                f"{grouped[session]} vs {values}"
            )
        grouped[session] = values
    return grouped


def build_folds(
    session_to_group: Dict[str, Tuple[str, ...]],
    folds_arg: int,
    exclude_sessions: Sequence[str],
) -> List[List[str]]:
    excluded = set(exclude_sessions)
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for session, key in session_to_group.items():
        if session in excluded:
            continue
        groups[key].append(session)
    for sessions in groups.values():
        sessions.sort()
    if not groups:
        raise RuntimeError("No sessions left after exclusions")

    min_group_size = min(len(sessions) for sessions in groups.values())
    folds = folds_arg if folds_arg > 0 else min_group_size
    if folds < 1:
        raise RuntimeError("fold count must be positive")
    if folds > min_group_size:
        undersized = {key: len(sessions) for key, sessions in groups.items() if len(sessions) < folds}
        print(
            f"[WARN] {folds} folds requested but {len(undersized)} behavior group(s) have fewer sessions "
            f"than that; those groups will be absent from validation in some folds (round-robin distributes "
            f"whatever sessions each group has):",
            flush=True,
        )
        for key, size in sorted(undersized.items()):
            print(f"  [WARN] group={key} sessions={size} -> missing from {folds - size} fold(s)", flush=True)

    fold_sessions: List[List[str]] = [[] for _ in range(folds)]
    for group_key in sorted(groups):
        sessions = groups[group_key]
        for idx, session in enumerate(sessions):
            fold_sessions[idx % folds].append(session)
    for sessions in fold_sessions:
        sessions.sort()
    return fold_sessions


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict], keys: Sequence[str]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for key in keys:
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.get(key, "")] += 1
        summary[key] = dict(sorted(counts.items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leak-free session-level CV train/val CSV splits."
    )
    parser.add_argument(
        "--source-csv",
        action="append",
        required=True,
        help="Source window CSV. Pass multiple times to combine current train and val CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for fold_XX_train.csv and fold_XX_val.csv.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Number of folds. Default: smallest behavior group size after exclusions.",
    )
    parser.add_argument(
        "--exclude-sessions",
        default="",
        help="Comma-separated sessions to omit from all folds, e.g. session_013.",
    )
    parser.add_argument(
        "--group-columns",
        default="material,non_slip_behavior,slip_behavior,slip_motion",
        help="Comma-separated columns defining CV strata/groups.",
    )
    args = parser.parse_args()

    rows, fieldnames = read_rows(args.source_csv)
    excluded = set(parse_list(args.exclude_sessions))
    group_columns = parse_list(args.group_columns)
    session_to_group = session_key(rows, group_columns)
    missing_excluded = sorted(excluded - set(session_to_group))
    if missing_excluded:
        raise RuntimeError(f"Excluded sessions not found: {missing_excluded}")

    fold_val_sessions = build_folds(session_to_group, args.folds, sorted(excluded))
    all_sessions = set(session_to_group) - excluded
    manifest = {
        "source_csv": args.source_csv,
        "out_dir": args.out_dir,
        "excluded_sessions": sorted(excluded),
        "group_columns": group_columns,
        "folds": [],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for fold_idx, val_sessions in enumerate(fold_val_sessions):
        val_set = set(val_sessions)
        train_set = all_sessions - val_set
        train_rows = [row for row in rows if row["session"] in train_set]
        val_rows = [row for row in rows if row["session"] in val_set]
        train_csv = os.path.join(args.out_dir, f"fold_{fold_idx:02d}_train.csv")
        val_csv = os.path.join(args.out_dir, f"fold_{fold_idx:02d}_val.csv")
        write_csv(train_csv, train_rows, fieldnames)
        write_csv(val_csv, val_rows, fieldnames)
        fold_info = {
            "fold": fold_idx,
            "train_csv": train_csv.replace("\\", "/"),
            "val_csv": val_csv.replace("\\", "/"),
            "train_sessions": sorted(train_set),
            "val_sessions": sorted(val_set),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "val_summary": summarize(val_rows, ["label", "session", "behavior", "motion_type"]),
        }
        manifest["folds"].append(fold_info)
        print(
            f"[fold {fold_idx:02d}] val_sessions={','.join(sorted(val_set))} "
            f"train_rows={len(train_rows)} val_rows={len(val_rows)}"
        )

    covered_val_sessions = [session for fold in manifest["folds"] for session in fold["val_sessions"]]
    missing_val_sessions = sorted(all_sessions - set(covered_val_sessions))
    duplicated_val_sessions = sorted(
        session for session in set(covered_val_sessions) if covered_val_sessions.count(session) > 1
    )
    if missing_val_sessions or duplicated_val_sessions:
        raise RuntimeError(
            "Invalid CV coverage: "
            f"missing_val_sessions={missing_val_sessions}, "
            f"duplicated_val_sessions={duplicated_val_sessions}"
        )
    print(f"[INFO] validation coverage: {len(covered_val_sessions)}/{len(all_sessions)} sessions")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

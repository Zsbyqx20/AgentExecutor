#!/usr/bin/env python3
"""Analyze average steps used in a benchmark checkpoint.

Filters tasks by source CSV keys, keeps successful tasks only, and resolves
trace files from one or more track roots.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average steps used from a checkpoint JSON, filtered by "
            "source CSV tasks and resolved track traces."
        )
    )
    parser.add_argument(
        "--track-dir",
        dest="track_dirs",
        action="append",
        required=True,
        help=(
            "Track root to scan. Repeat the flag for multiple roots, "
            "e.g. --track-dir tracks --track-dir tracks-doubao"
        ),
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Source benchmark CSV path (must contain env_id and task_id columns).",
    )
    parser.add_argument(
        "--input-ckpt",
        required=True,
        help="Checkpoint JSON path.",
    )
    return parser.parse_args()


def parse_ckpt_key(key_str: str) -> tuple[str, int] | None:
    try:
        parsed = ast.literal_eval(key_str)
    except (ValueError, SyntaxError):
        return None

    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 2
        or not isinstance(parsed[0], str)
        or not isinstance(parsed[1], int)
    ):
        return None
    return parsed[0], parsed[1]


def load_source_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        required = {"env_id", "task_id"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV missing required columns ({missing_str}): {csv_path}")

        keys: set[tuple[str, int]] = set()
        for row in reader:
            env_id = str(row["env_id"])
            task_id_raw = row["task_id"]
            try:
                task_id = int(task_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid task_id in CSV row {row!r}: expected integer-like value"
                ) from exc
            keys.add((env_id, task_id))
    return keys


def load_ckpt(ckpt_path: Path) -> dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint JSON not found: {ckpt_path}")

    with ckpt_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected checkpoint JSON top-level dict, got {type(payload).__name__}: {ckpt_path}"
        )
    return payload


def resolve_trace_path(track_dirs: list[Path], worker_id: str, track: str) -> Path | None:
    for track_dir in track_dirs:
        candidate = track_dir / worker_id / track / "trace.json"
        if candidate.exists():
            return candidate
    return None


def extract_step_count(trace_path: Path) -> int | None:
    try:
        with trace_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None

    return len(steps)


def analyze(
    source_keys: set[tuple[str, int]],
    ckpt_payload: dict[str, Any],
    track_dirs: list[Path],
    ckpt_path: Path,
) -> dict[str, Any]:
    matched_count = 0
    success_count = 0
    resolved_count = 0
    missing_trace_count = 0
    invalid_record_count = 0
    step_counts: list[int] = []

    for key_str, record in ckpt_payload.items():
        key = parse_ckpt_key(key_str)
        if key is None:
            invalid_record_count += 1
            continue

        if key not in source_keys:
            continue
        matched_count += 1

        if not isinstance(record, dict):
            invalid_record_count += 1
            continue

        if record.get("success") is not True:
            continue
        success_count += 1

        worker_id = record.get("worker_id")
        track = record.get("track")
        if not isinstance(worker_id, str) or not worker_id:
            invalid_record_count += 1
            continue
        if not isinstance(track, str) or not track:
            invalid_record_count += 1
            continue

        trace_path = resolve_trace_path(track_dirs, worker_id, track)
        if trace_path is None:
            missing_trace_count += 1
            continue

        step_count = extract_step_count(trace_path)
        if step_count is None:
            invalid_record_count += 1
            continue

        step_counts.append(step_count)
        resolved_count += 1

    average_steps = statistics.fmean(step_counts) if step_counts else None
    median_steps = statistics.median(step_counts) if step_counts else None

    return {
        "input_ckpt": str(ckpt_path),
        "input_csv_task_count": len(source_keys),
        "matched_ckpt_entries": matched_count,
        "success_filtered_entries": success_count,
        "trace_resolved_entries": resolved_count,
        "missing_trace_count": missing_trace_count,
        "invalid_record_count": invalid_record_count,
        "average_steps": average_steps,
        "min_steps": min(step_counts) if step_counts else None,
        "max_steps": max(step_counts) if step_counts else None,
        "median_steps": median_steps,
    }


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_report(result: dict[str, Any]) -> None:
    print(f"input_ckpt: {result['input_ckpt']}")
    print(f"input_csv_task_count: {result['input_csv_task_count']}")
    print(f"matched_ckpt_entries: {result['matched_ckpt_entries']}")
    print(f"success_filtered_entries: {result['success_filtered_entries']}")
    print(f"trace_resolved_entries: {result['trace_resolved_entries']}")
    print(f"missing_trace_count: {result['missing_trace_count']}")
    print(f"invalid_record_count: {result['invalid_record_count']}")
    print(f"average_steps: {_fmt_num(result['average_steps'])}")
    print(f"min_steps: {_fmt_num(result['min_steps'])}")
    print(f"max_steps: {_fmt_num(result['max_steps'])}")
    print(f"median_steps: {_fmt_num(result['median_steps'])}")


def main() -> None:
    args = parse_args()
    track_dirs = [Path(p) for p in args.track_dirs]
    source_keys = load_source_keys(Path(args.input_csv))
    ckpt_payload = load_ckpt(Path(args.input_ckpt))
    result = analyze(source_keys, ckpt_payload, track_dirs, Path(args.input_ckpt))
    print_report(result)


if __name__ == "__main__":
    main()

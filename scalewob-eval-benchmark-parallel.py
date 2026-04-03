import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from scalewob import ScaleWoBAutomation

from age import (
    BrownieScaleWoBAgent,
    DoubaoScaleWoBAgent,
    GeneralScaleWoBAgent,
    MAIScaleWoBAgent,
    MobileAgent35ScaleWoBAgent,
    OpenAutoGLMScaleWoBAgent,
    RGMScaleWoBAgent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScaleWoB benchmark tasks in parallel."
    )
    parser.add_argument(
        "--input-csv",
        default="data/benchmark_source_20260210.csv",
        help="Input benchmark CSV path.",
    )
    parser.add_argument(
        "--output-json",
        default="data/benchmark_eval_20260210.json",
        help="Merged checkpoint JSON path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to auto policy.",
    )
    parser.add_argument(
        "--max-workers-cap",
        type=int,
        default=4,
        help="Upper bound used by auto worker policy.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser automation in headless mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Agent max steps per task.",
    )
    parser.add_argument(
        "--tracks-dir",
        default="tracks",
        help="Base directory for tracking outputs.",
    )
    parser.add_argument(
        "--worker-shards-dir",
        default="data/.benchmark_shards",
        help="Directory for per-worker shard files.",
    )
    parser.add_argument(
        "--model-name",
        default="doubao-seed-1-8-251228",
        help="Model name passed to the selected agent.",
    )
    parser.add_argument(
        "--agent",
        choices=[
            "doubao",
            "brownie",
            "general",
            "open-autoglm",
            "mobileagent35",
            "mai",
            "rgm",
        ],
        default="doubao",
        help="Agent backend to use.",
    )
    parser.add_argument(
        "--no-param-task",
        action="store_true",
        help="Skip tasks whose ScaleWoB task params are not None.",
    )
    parser.add_argument(
        "--rerun-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to rerun checkpoint records with status="failed".',
    )
    parser.add_argument(
        "--vertex-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Vertex AI API client instead of OpenAI-compatible HTTP endpoint.",
    )
    return parser.parse_args()


def now_iso_utc() -> str:
    return datetime.now(UTC).isoformat()


def make_key(env_id: str, task_id: int) -> tuple[str, int]:
    return env_id, int(task_id)


def make_key_string(env_id: str, task_id: int) -> str:
    return str(make_key(env_id, task_id))


def load_json_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def parse_updated_at(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        else:
            return parsed
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def should_replace(
    old_record: dict[str, Any] | None, new_record: dict[str, Any]
) -> bool:
    if old_record is None:
        return True
    return parse_updated_at(new_record.get("updated_at")) >= parse_updated_at(
        old_record.get("updated_at")
    )


def write_output_json(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def append_shard(shard_path: Path, entry: dict[str, Any]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def merge_shards(
    merged: dict[str, dict[str, Any]], shard_dir: Path
) -> dict[str, dict[str, Any]]:
    if not shard_dir.exists():
        return merged

    out = dict(merged)
    for shard in sorted(shard_dir.glob("worker_*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = payload.get("key")
                record = payload.get("record")
                if not isinstance(key, str) or not isinstance(record, dict):
                    continue
                if should_replace(out.get(key), record):
                    out[key] = record
    return out


def failure_record(
    row: dict[str, Any],
    *,
    task_params: dict[str, Any] | None,
    error: Exception,
    worker_id: str,
) -> dict[str, Any]:
    return {
        "task_type": row.get("task_type"),
        "task_params": task_params,
        "lang": row.get("lang"),
        "description": row.get("description"),
        "cn_description": row.get("cn_description"),
        "track": None,
        "success": False,
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "worker_id": worker_id,
        "updated_at": now_iso_utc(),
    }


def build_agent(
    agent_name: str,
    *,
    model_name: str,
    task_name: str,
    lang: str,
    env_id: str,
    task_id: int,
    task_params: dict[str, Any] | None,
    headless: bool,
    tracks_dir: str,
    worker_id: str,
    max_steps: int,
    use_vertex_api: bool,
) -> Any:
    agent_map = {
        "doubao": DoubaoScaleWoBAgent,
        "brownie": BrownieScaleWoBAgent,
        "general": GeneralScaleWoBAgent,
        "open-autoglm": OpenAutoGLMScaleWoBAgent,
        "mobileagent35": MobileAgent35ScaleWoBAgent,
        "mai": MAIScaleWoBAgent,
        "rgm": RGMScaleWoBAgent,
    }
    agent_cls = agent_map.get(agent_name)
    if agent_cls is None:
        raise ValueError(f"Unsupported agent: {agent_name}")
    return agent_cls(
        model_name=model_name,
        task_name=task_name,
        language=lang,
        env_id=env_id,
        task_id=task_id,
        task_params=task_params,
        headless=headless,
        platform="mobile",
        enable_tracking=True,
        tracking_output_dir=str(Path(tracks_dir) / worker_id),
        max_steps=max_steps,
        use_vertex_api=use_vertex_api,
    )


def run_task(
    row: dict[str, Any],
    *,
    agent_name: str,
    model_name: str,
    headless: bool,
    max_steps: int,
    tracks_dir: str,
    worker_shards_dir: str,
    use_vertex_api: bool,
) -> dict[str, Any]:
    env_id = str(row["env_id"])
    task_id = int(row["task_id"])
    key = make_key_string(env_id, task_id)
    pid = os.getpid()
    worker_id = f"worker_{pid}"
    shard_path = Path(worker_shards_dir) / f"worker_{pid}.jsonl"

    result: dict[str, Any]
    task_params: dict[str, Any] | None = None

    try:
        with ScaleWoBAutomation(env_id=env_id, headless=headless) as auto:
            tasks = auto.tasks

        matched_tasks = [t for t in tasks if int(t["task_id"]) == task_id]
        if len(matched_tasks) != 1:
            raise ValueError(
                f"Expected one task match for [{env_id}:{task_id}], got {len(matched_tasks)}"
            )

        task_params = matched_tasks[0].get("params")
        lang = "CN" if row.get("lang") == "中文" else "EN"
        task_name = row.get("description")
        if not task_name:
            task_name = row.get("description") or row.get("cn_description") or ""

        with build_agent(
            agent_name,
            model_name=model_name,
            task_name=task_name,
            lang=lang,
            env_id=env_id,
            task_id=task_id,
            task_params=task_params,
            headless=headless,
            tracks_dir=tracks_dir,
            worker_id=worker_id,
            max_steps=max_steps,
            use_vertex_api=use_vertex_api,
        ) as agent:
            while True:
                action = agent.predict()
                if action.get("action_type") == "finished":
                    result = {
                        "task_type": row.get("task_type"),
                        "task_params": task_params,
                        "lang": row.get("lang"),
                        "description": row.get("description"),
                        "cn_description": row.get("cn_description"),
                        "track": agent.tracker.task_dir.stem if agent.tracker else None,
                        "success": bool(agent.task_status),
                        "status": "completed",
                        "error_type": None,
                        "error_message": None,
                        "worker_id": worker_id,
                        "updated_at": now_iso_utc(),
                    }
                    break
    except Exception as exc:  # noqa: BLE001
        result = failure_record(
            row,
            task_params=task_params,
            error=exc,
            worker_id=worker_id,
        )

    entry = {"key": key, "record": result}
    append_shard(shard_path, entry)
    return entry


def determine_workers(requested: int | None, pending_count: int, cap: int) -> int:
    if requested is not None:
        return max(1, requested)
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, pending_count, cap))


def is_failed_record(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    return record.get("status") == "failed"


def build_pending_rows(
    rows: list[dict[str, Any]],
    merged_results: dict[str, dict[str, Any]],
    *,
    rerun_failed: bool,
) -> tuple[list[dict[str, Any]], int, int, int]:
    pending_rows: list[dict[str, Any]] = []
    missing_count = 0
    failed_retry_count = 0
    skipped_count = 0

    for row in rows:
        key = make_key_string(str(row["env_id"]), int(row["task_id"]))
        existing = merged_results.get(key)

        if existing is None:
            pending_rows.append(row)
            missing_count += 1
            continue

        if rerun_failed and is_failed_record(existing):
            pending_rows.append(row)
            failed_retry_count += 1
            continue

        skipped_count += 1

    return pending_rows, missing_count, failed_retry_count, skipped_count


def parse_has_params(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f", ""}:
            return False
    raise ValueError(f"Invalid has_params value: {value!r}")


def main() -> None:
    args = parse_args()
    console = Console()

    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    shard_dir = Path(args.worker_shards_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    rows = df.to_dict(orient="records")

    merged_results = load_json_map(output_json)
    merged_results = merge_shards(merged_results, shard_dir)
    write_output_json(output_json, merged_results)

    pending_rows, missing_count, failed_retry_count, skipped_count = build_pending_rows(
        rows,  # type: ignore
        merged_results,
        rerun_failed=args.rerun_failed,
    )

    param_filtered_count = 0
    if args.no_param_task and pending_rows:
        if "has_params" not in df.columns:
            raise ValueError(
                "Input CSV is missing required column 'has_params' for --no-param-task. "
                "Please patch/generate has_params in the source CSV first."
            )
        filtered_rows: list[dict[str, Any]] = []
        for row in pending_rows:
            if parse_has_params(row.get("has_params")):
                param_filtered_count += 1
                continue
            filtered_rows.append(row)
        pending_rows = filtered_rows

    total_rows = len(rows)
    pending_count = len(pending_rows)
    workers = determine_workers(args.workers, pending_count, args.max_workers_cap)

    console.print(
        "Loaded "
        f"{total_rows} tasks, "
        f"{len(merged_results)} in checkpoint, "
        f"{missing_count} missing, "
        f"{failed_retry_count} failed retries, "
        f"{skipped_count} skipped, "
        f"{param_filtered_count} skipped by --no-param-task, "
        f"{pending_count} queued."
    )
    if pending_count == 0:
        success = sum(1 for item in merged_results.values() if item.get("success"))
        rate = (
            round((success * 100 / len(merged_results)), 2) if merged_results else 0.0
        )
        console.print(f"All tasks done! Success rate: {rate}%")
        return

    console.print(f"Starting parallel run with {workers} worker(s)...")
    console.print(
        f"API backend: {'vertex' if args.vertex_api else 'openai-compatible'}"
    )

    futures = {}
    completed_since_sync = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in pending_rows:
            future = executor.submit(
                run_task,
                row,
                agent_name=args.agent,
                model_name=args.model_name,
                headless=args.headless,
                max_steps=args.max_steps,
                tracks_dir=args.tracks_dir,
                worker_shards_dir=args.worker_shards_dir,
                use_vertex_api=args.vertex_api,
            )
            futures[future] = row

        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            env_id = row.get("env_id")
            task_id = row.get("task_id")

            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001
                key = make_key_string(str(env_id), int(task_id))
                entry = {
                    "key": key,
                    "record": failure_record(
                        row,
                        task_params=None,
                        error=exc,
                        worker_id="parent",
                    ),
                }

            key = entry["key"]
            record = entry["record"]
            if should_replace(merged_results.get(key), record):
                merged_results[key] = record

            completed_since_sync += 1
            if completed_since_sync >= 10:
                merged_results = merge_shards(merged_results, shard_dir)
                write_output_json(output_json, merged_results)
                completed_since_sync = 0

            status = record.get("status")
            success = record.get("success")
            console.print(
                f"[{index}/{pending_count}] {env_id}:{task_id} -> {status}, success={success}"
            )

    merged_results = merge_shards(merged_results, shard_dir)
    write_output_json(output_json, merged_results)

    success_count = sum(1 for item in merged_results.values() if item.get("success"))
    total_count = len(merged_results)
    success_rate = round((success_count * 100 / total_count), 2) if total_count else 0.0
    console.print(
        f"All tasks done! Success rate: {success_rate}% ({success_count}/{total_count})"
    )


if __name__ == "__main__":
    main()

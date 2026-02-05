#!/usr/bin/env python3
"""Analyze cost, token usage, action counts, and API calls for trajectory outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_ROOT = "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_float(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0


def _to_int(value: Any) -> int:
    return int(value) if _is_number(value) else 0


def _iter_trajectory_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.traj.json"))
    raise FileNotFoundError(f"Path does not exist: {root}")


def _count_role(messages: Any, role: str) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(1 for message in messages if isinstance(message, dict) and message.get("role") == role)


def _iter_usage_dicts(message: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    usage = message.get("usage")
    if isinstance(usage, dict):
        yield usage

    response = message.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            yield usage

    extra = message.get("extra")
    if isinstance(extra, dict):
        usage = extra.get("usage")
        if isinstance(usage, dict):
            yield usage
        response = extra.get("response")
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                yield usage


def _sum_usage_from_messages(messages: Any) -> Dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "count": 0,
    }
    if not isinstance(messages, list):
        return totals

    for message in messages:
        if not isinstance(message, dict):
            continue
        for usage in _iter_usage_dicts(message):
            totals["count"] += 1
            prompt_tokens = _to_int(usage.get("prompt_tokens", 0))
            completion_tokens = _to_int(usage.get("completion_tokens", 0))
            total_tokens = _to_int(usage.get("total_tokens", 0))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["total_tokens"] += total_tokens
            totals["cost"] += _to_float(usage.get("cost", 0.0))

    return totals


def _extract_model_stats(obj: Dict[str, Any]) -> Dict[str, Any]:
    info = obj.get("info")
    if isinstance(info, dict):
        model_stats = info.get("model_stats")
        if isinstance(model_stats, dict):
            return model_stats
    model_stats = obj.get("model_stats")
    if isinstance(model_stats, dict):
        return model_stats
    return {}


def _extract_cost(obj: Dict[str, Any], usage_cost: float) -> float:
    model_stats = _extract_model_stats(obj)
    for key in ("total_cost", "instance_cost", "cost"):
        value = model_stats.get(key)
        if _is_number(value):
            return float(value)
    for key in ("total_cost", "instance_cost", "cost"):
        value = obj.get(key)
        if _is_number(value):
            return float(value)
    return float(usage_cost)


def _extract_tokens(obj: Dict[str, Any], usage_totals: Dict[str, Any]) -> Dict[str, int]:
    model_stats = _extract_model_stats(obj)
    tokens_sent = _to_int(model_stats.get("tokens_sent", 0))
    tokens_received = _to_int(model_stats.get("tokens_received", 0))
    total_tokens = _to_int(model_stats.get("total_tokens", 0))
    if total_tokens == 0 and (tokens_sent or tokens_received):
        total_tokens = tokens_sent + tokens_received
    if total_tokens == 0 and tokens_sent == 0 and tokens_received == 0:
        tokens_sent = _to_int(usage_totals.get("prompt_tokens", 0))
        tokens_received = _to_int(usage_totals.get("completion_tokens", 0))
        total_tokens = _to_int(usage_totals.get("total_tokens", 0))
    return {
        "prompt_tokens": tokens_sent,
        "completion_tokens": tokens_received,
        "total_tokens": total_tokens,
    }


def _extract_api_calls(
    obj: Dict[str, Any],
    *,
    usage_count: int,
    assistant_count: int,
    tool_count: int,
) -> int:
    model_stats = _extract_model_stats(obj)
    for key in ("api_calls", "n_calls", "num_calls", "call_count"):
        value = model_stats.get(key)
        if _is_number(value):
            return int(value)
    for key in ("api_calls", "n_calls", "num_calls", "call_count"):
        value = obj.get(key)
        if _is_number(value):
            return int(value)
    if usage_count:
        return int(usage_count)
    if assistant_count:
        return int(assistant_count)
    if tool_count:
        return int(tool_count)
    return 0


def _extract_actions(
    obj: Dict[str, Any],
    *,
    assistant_count: int,
    tool_count: int,
    usage_count: int,
    api_calls: int,
) -> int:
    for key in ("actions", "num_actions", "action_count", "n_actions"):
        value = obj.get(key)
        if _is_number(value):
            return int(value)

    trajectory = obj.get("trajectory")
    if isinstance(trajectory, list):
        return len(trajectory)

    for key in ("steps", "num_steps", "n_steps", "step_count", "total_steps"):
        value = obj.get(key)
        if _is_number(value):
            return int(value)

    if assistant_count:
        return int(assistant_count)
    if tool_count:
        return int(tool_count)
    if usage_count:
        return int(usage_count)
    return int(api_calls)


def _guess_instance_id(path: Path, obj: Dict[str, Any]) -> str | None:
    instance_id = obj.get("instance_id")
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    name = path.name
    if name.endswith(".traj.json"):
        return name[: -len(".traj.json")]
    return path.stem


def _group_key(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name or "."
    return rel.parts[0] if rel.parts else "."


def _empty_metrics() -> Dict[str, Any]:
    return {
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "actions": 0,
        "api_calls": 0,
    }


def _merge_metrics(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["cost"] += source.get("cost", 0.0)
    target["prompt_tokens"] += source.get("prompt_tokens", 0)
    target["completion_tokens"] += source.get("completion_tokens", 0)
    target["total_tokens"] += source.get("total_tokens", 0)
    target["actions"] += source.get("actions", 0)
    target["api_calls"] += source.get("api_calls", 0)


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _analyze_payload(obj: Dict[str, Any], path: Path, *, root: Path) -> Dict[str, Any]:
    messages = obj.get("messages", [])
    usage_totals = _sum_usage_from_messages(messages)

    assistant_count = _count_role(messages, "assistant")
    tool_count = _count_role(messages, "tool")

    api_calls = _extract_api_calls(
        obj,
        usage_count=usage_totals["count"],
        assistant_count=assistant_count,
        tool_count=tool_count,
    )
    actions = _extract_actions(
        obj,
        assistant_count=assistant_count,
        tool_count=tool_count,
        usage_count=usage_totals["count"],
        api_calls=api_calls,
    )
    tokens = _extract_tokens(obj, usage_totals)
    cost = _extract_cost(obj, usage_totals["cost"])

    return {
        "path": _display_path(root, path),
        "instance_id": _guess_instance_id(path, obj),
        "cost": cost,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "actions": actions,
        "api_calls": api_calls,
    }


def summarize(root: Path, *, include_files: bool = False) -> Dict[str, Any]:
    files = _iter_trajectory_files(root)
    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "files_count": len(files),
        "errors": 0,
        "totals": _empty_metrics(),
        "runs": [],
    }

    runs: Dict[str, Dict[str, Any]] = {}

    for path in files:
        group = _group_key(root, path)
        if group not in runs:
            run_entry: Dict[str, Any] = {
                "run_dir": group,
                "files_count": 0,
                "errors": 0,
                "totals": _empty_metrics(),
            }
            if include_files:
                run_entry["files"] = []
            runs[group] = run_entry
        run_entry = runs[group]
        run_entry["files_count"] += 1

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                obj: Dict[str, Any] = {"messages": payload}
            elif isinstance(payload, dict):
                obj = payload
            else:
                raise ValueError("Unexpected JSON root type")
            metrics = _analyze_payload(obj, path, root=root)
        except (OSError, json.JSONDecodeError, ValueError):
            summary["errors"] += 1
            run_entry["errors"] += 1
            continue

        _merge_metrics(summary["totals"], metrics)
        _merge_metrics(run_entry["totals"], metrics)
        if include_files and isinstance(run_entry["files"], list):
            run_entry["files"].append(metrics)

    def _average(metrics: Dict[str, Any], denom: int) -> Dict[str, Any]:
        if denom <= 0:
            return {key: 0.0 for key in metrics}
        return {key: metrics[key] / denom for key in metrics}

    summary["averages_per_file"] = _average(summary["totals"], summary["files_count"])

    for run_entry in runs.values():
        run_entry["averages_per_file"] = _average(run_entry["totals"], run_entry["files_count"])
        summary["runs"].append(run_entry)

    summary["runs"] = sorted(summary["runs"], key=lambda item: item["run_dir"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize cost, tokens, action count, and API calls from .traj.json outputs.")
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file. If omitted, prints JSON to stdout.",
    )
    parser.add_argument(
        "--include-files",
        action="store_true",
        help="Include per-file metrics in the output.",
    )
    args = parser.parse_args()

    summary = summarize(Path(args.root), include_files=args.include_files)
    payload = json.dumps(summary, indent=2, sort_keys=False)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

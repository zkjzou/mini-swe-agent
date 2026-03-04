#!/usr/bin/env python3
"""Analyze cost, token usage, action counts, and API calls for trajectory outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_ROOT = "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM"
_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


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


def _parse_models(models: list[str] | None) -> set[str]:
    parsed: set[str] = set()
    if not models:
        return parsed
    for item in models:
        for token in item.split(","):
            token = token.strip()
            if token:
                parsed.add(token)
    return parsed


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


def _empty_usage_totals() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "count": 0,
    }


def _usage_to_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens = _to_int(usage.get("prompt_tokens", 0))
    completion_tokens = _to_int(usage.get("completion_tokens", 0))
    total_tokens = _to_int(usage.get("total_tokens", 0))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _add_usage_dict(totals: Dict[str, Any], usage: Dict[str, Any]) -> None:
    totals["count"] += 1
    usage_tokens = _usage_to_tokens(usage)
    totals["prompt_tokens"] += usage_tokens["prompt_tokens"]
    totals["completion_tokens"] += usage_tokens["completion_tokens"]
    totals["total_tokens"] += usage_tokens["total_tokens"]
    totals["cost"] += _to_float(usage.get("cost", 0.0))


def _merge_usage_totals(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["prompt_tokens"] += _to_int(source.get("prompt_tokens", 0))
    target["completion_tokens"] += _to_int(source.get("completion_tokens", 0))
    target["total_tokens"] += _to_int(source.get("total_tokens", 0))
    target["cost"] += _to_float(source.get("cost", 0.0))
    target["count"] += _to_int(source.get("count", 0))


def _sum_usage_from_message(message: Any) -> Dict[str, Any]:
    totals = _empty_usage_totals()
    if not isinstance(message, dict):
        return totals
    for usage in _iter_usage_dicts(message):
        _add_usage_dict(totals, usage)
    return totals


def _sum_usage_from_messages(messages: Any) -> Dict[str, Any]:
    totals = _empty_usage_totals()
    if not isinstance(messages, list):
        return totals

    for message in messages:
        _merge_usage_totals(totals, _sum_usage_from_message(message))

    return totals


def _iter_verifier_response_messages(message: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return

    verifier = extra.get("verifier")
    if not isinstance(verifier, dict):
        return

    verifier_output = verifier.get("verifier_output")
    if not isinstance(verifier_output, dict):
        return

    response = verifier_output.get("response")
    if isinstance(response, dict):
        yield response

    responses = verifier_output.get("responses")
    if isinstance(responses, list):
        for item in responses:
            if isinstance(item, dict):
                yield item

    checklist = verifier_output.get("checklist")
    if isinstance(checklist, dict):
        checklist_response = checklist.get("response")
        if isinstance(checklist_response, dict):
            yield checklist_response


def _sum_verifier_usage_from_messages(messages: Any) -> Dict[str, Any]:
    totals = _empty_usage_totals()
    if not isinstance(messages, list):
        return totals
    for message in messages:
        if not isinstance(message, dict):
            continue
        for verifier_message in _iter_verifier_response_messages(message):
            _merge_usage_totals(totals, _sum_usage_from_message(verifier_message))
    return totals


def _reconcile_token_split(
    *,
    aggregate_tokens: Dict[str, int],
    agent_usage: Dict[str, Any],
    verifier_usage: Dict[str, Any],
) -> Dict[str, int]:
    split_tokens: Dict[str, int] = {}
    for token_key in _TOKEN_KEYS:
        target_total = max(0, _to_int(aggregate_tokens.get(token_key, 0)))
        agent_value = max(0, _to_int(agent_usage.get(token_key, 0)))
        verifier_value = max(0, _to_int(verifier_usage.get(token_key, 0)))
        raw_total = agent_value + verifier_value

        if raw_total == target_total:
            reconciled_agent = agent_value
            reconciled_verifier = verifier_value
        elif target_total == 0:
            reconciled_agent = 0
            reconciled_verifier = 0
        elif raw_total == 0:
            reconciled_agent = target_total
            reconciled_verifier = 0
        else:
            verifier_share = verifier_value / raw_total
            reconciled_verifier = int(round(target_total * verifier_share))
            reconciled_verifier = max(0, min(reconciled_verifier, target_total))
            reconciled_agent = target_total - reconciled_verifier

        split_tokens[f"agent_{token_key}"] = reconciled_agent
        split_tokens[f"verifier_{token_key}"] = reconciled_verifier
    return split_tokens


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


def _filter_files_by_groups(root: Path, files: list[Path], groups: set[str]) -> tuple[list[Path], list[str]]:
    if not groups:
        return files, []
    available = {_group_key(root, path) for path in files}
    missing = sorted(groups - available)
    filtered = [path for path in files if _group_key(root, path) in groups]
    return filtered, missing


def _empty_metrics() -> Dict[str, Any]:
    return {
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "agent_prompt_tokens": 0,
        "agent_completion_tokens": 0,
        "agent_total_tokens": 0,
        "verifier_prompt_tokens": 0,
        "verifier_completion_tokens": 0,
        "verifier_total_tokens": 0,
        "actions": 0,
        "api_calls": 0,
    }


def _merge_metrics(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["cost"] += source.get("cost", 0.0)
    target["prompt_tokens"] += source.get("prompt_tokens", 0)
    target["completion_tokens"] += source.get("completion_tokens", 0)
    target["total_tokens"] += source.get("total_tokens", 0)
    target["agent_prompt_tokens"] += source.get("agent_prompt_tokens", 0)
    target["agent_completion_tokens"] += source.get("agent_completion_tokens", 0)
    target["agent_total_tokens"] += source.get("agent_total_tokens", 0)
    target["verifier_prompt_tokens"] += source.get("verifier_prompt_tokens", 0)
    target["verifier_completion_tokens"] += source.get("verifier_completion_tokens", 0)
    target["verifier_total_tokens"] += source.get("verifier_total_tokens", 0)
    target["actions"] += source.get("actions", 0)
    target["api_calls"] += source.get("api_calls", 0)


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _analyze_payload(obj: Dict[str, Any], path: Path, *, root: Path) -> Dict[str, Any]:
    messages = obj.get("messages", [])
    agent_usage_totals = _sum_usage_from_messages(messages)
    verifier_usage_totals = _sum_verifier_usage_from_messages(messages)
    usage_totals = _empty_usage_totals()
    _merge_usage_totals(usage_totals, agent_usage_totals)
    _merge_usage_totals(usage_totals, verifier_usage_totals)

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
    split_tokens = _reconcile_token_split(
        aggregate_tokens=tokens,
        agent_usage=agent_usage_totals,
        verifier_usage=verifier_usage_totals,
    )

    return {
        "path": _display_path(root, path),
        "instance_id": _guess_instance_id(path, obj),
        "cost": cost,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "agent_prompt_tokens": split_tokens["agent_prompt_tokens"],
        "agent_completion_tokens": split_tokens["agent_completion_tokens"],
        "agent_total_tokens": split_tokens["agent_total_tokens"],
        "verifier_prompt_tokens": split_tokens["verifier_prompt_tokens"],
        "verifier_completion_tokens": split_tokens["verifier_completion_tokens"],
        "verifier_total_tokens": split_tokens["verifier_total_tokens"],
        "actions": actions,
        "api_calls": api_calls,
    }


def summarize(root: Path, *, include_files: bool = False, models: set[str] | None = None) -> Dict[str, Any]:
    files = _iter_trajectory_files(root)
    selected_models = set(models or [])
    files, missing_models = _filter_files_by_groups(root, files, selected_models)
    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "files_count": len(files),
        "errors": 0,
        "totals": _empty_metrics(),
        "runs": [],
    }
    if selected_models:
        summary["selected_models"] = sorted(selected_models)
        if missing_models:
            summary["missing_models"] = missing_models

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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Exact top-level run_dir names to include (space- and/or comma-separated). "
            "Example: --models run_a run_b"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    models = _parse_models(args.models)
    if root.is_file() and models:
        raise ValueError("--models requires --root to be a directory")

    summary = summarize(root, include_files=args.include_files, models=models)
    missing_models = summary.get("missing_models", [])
    if missing_models:
        raise ValueError(f"Exact model match not found for: {', '.join(missing_models)}")
    payload = json.dumps(summary, indent=2, sort_keys=False)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

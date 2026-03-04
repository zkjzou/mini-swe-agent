from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Literal

from minisweagent.utils.verifier_action_sampling import DATASET_VERSION

DedupeMode = Literal["none", "exact", "semantic_key"]
ConflictPolicy = Literal["keep_first", "keep_last", "error"]
Record = tuple[dict[str, Any], str, int]
GroupKey = tuple[Any, ...]


def _iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError("No input JSONL files found.")
    return deduped


def _canonical_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("instance_id"),
        row.get("run_id"),
        row.get("trajectory_relpath"),
        row.get("step_index"),
        row.get("message_index"),
        row.get("candidate_source"),
        row.get("sampler_model_id"),
        row.get("sample_index"),
    )


def _group_key(row: dict[str, Any]) -> GroupKey:
    return (
        row.get("instance_id"),
        row.get("run_id"),
        row.get("trajectory_relpath"),
        row.get("step_index"),
        row.get("message_index"),
    )


def _step_key(row: dict[str, Any]) -> GroupKey:
    return _group_key(row)


def _normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _rows_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _normalized_json(a) == _normalized_json(b)


def _parse_int(value: Any, default: int = -1) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _group_sort_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    instance_id = str(row.get("instance_id") or "")
    run_id = str(row.get("run_id") or "")
    step_index = _parse_int(row.get("step_index"), default=-1)
    message_index = _parse_int(row.get("message_index"), default=-1)
    trajectory_relpath = str(row.get("trajectory_relpath") or "")
    return (instance_id, run_id, step_index, message_index, trajectory_relpath)


def _first_valid_action(actions: Any) -> dict[str, Any] | None:
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, dict):
            continue
        command = action.get("command")
        if not isinstance(command, str) or not command:
            continue
        return copy.deepcopy(action)
    return None


def _resolve_record_conflict(
    *,
    existing: Record,
    incoming: Record,
    key: tuple[Any, ...],
    conflict_policy: ConflictPolicy,
    counts: dict[str, int],
    conflicts: list[dict[str, Any]],
    stage: str,
) -> Record:
    existing_row, _, _ = existing
    incoming_row, incoming_file, incoming_line = incoming

    if _rows_equal(existing_row, incoming_row):
        counts["duplicates_dropped"] += 1
        return existing

    counts["conflicts"] += 1
    conflicts.append(
        {
            "stage": stage,
            "input_file": incoming_file,
            "line": incoming_line,
            "key": key,
        }
    )

    if conflict_policy == "keep_first":
        return existing
    if conflict_policy == "keep_last":
        return incoming
    if conflict_policy == "error":
        raise ValueError(
            f"Conflict for key {key} between existing row and {incoming_file}:{incoming_line} "
            "(set --conflict-policy keep_first|keep_last to resolve automatically)."
        )
    raise ValueError(f"Unknown conflict policy '{conflict_policy}'.")


def _row_to_slot_key(row: dict[str, Any]) -> tuple[str, str, int | None] | None:
    source = str(row.get("candidate_source") or "")
    if source == "gold":
        return ("gold", "gold", None)
    if source != "sampled":
        return None
    sampler_model_id = str(row.get("sampler_model_id") or "").strip()
    if not sampler_model_id:
        return None
    sample_index = _parse_int(row.get("sample_index"), default=0)
    if sample_index != 0:
        return None
    return ("sampled", sampler_model_id, 0)


def _record_to_public_row(record: Record) -> dict[str, Any]:
    return copy.deepcopy(record[0])


def merge_verifier_sampling_datasets(
    *,
    input_paths: list[Path],
    output_jsonl: Path,
    output_summary: Path | None = None,
    dedupe: DedupeMode = "semantic_key",
    conflict_policy: ConflictPolicy = "keep_first",
    require_gold: bool = True,
    sort_by_instance_step_model_sample: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if dedupe not in {"none", "exact", "semantic_key"}:
        raise ValueError(f"Unknown dedupe mode '{dedupe}'.")
    if conflict_policy not in {"keep_first", "keep_last", "error"}:
        raise ValueError(f"Unknown conflict policy '{conflict_policy}'.")
    if output_summary is None:
        output_summary = output_jsonl.with_name("merged_summary.json")
    if not overwrite and output_jsonl.exists():
        raise FileExistsError(f"Output file already exists: {output_jsonl}")
    if not overwrite and output_summary.exists():
        raise FileExistsError(f"Summary file already exists: {output_summary}")

    input_files = _iter_input_files(input_paths)

    counts: dict[str, int] = {
        "input_files": len(input_files),
        "input_rows": 0,
        "parsed_rows": 0,
        "invalid_rows": 0,
        "duplicates_dropped": 0,
        "conflicts": 0,
        "missing_gold_steps": 0,
        "missing_gold_rows_dropped": 0,
        "nonzero_sample_rows_ignored": 0,
        "groups_total": 0,
        "groups_dropped_missing_gold": 0,
        "actions_kept_total": 0,
        "actions_kept_gold": 0,
        "actions_kept_sampled": 0,
        "groups_with_5_actions": 0,
    }
    conflicts: list[dict[str, Any]] = []

    parsed_records: list[Record] = []
    exact_seen: set[str] = set()
    semantic_seen: dict[tuple[Any, ...], Record] = {}
    semantic_order: list[tuple[Any, ...]] = []

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                counts["input_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["invalid_rows"] += 1
                    continue
                if not isinstance(row, dict):
                    counts["invalid_rows"] += 1
                    continue
                counts["parsed_rows"] += 1

                incoming: Record = (row, str(input_file), line_no)

                if dedupe == "none":
                    parsed_records.append(incoming)
                    continue

                if dedupe == "exact":
                    row_hash = _normalized_json(row)
                    if row_hash in exact_seen:
                        counts["duplicates_dropped"] += 1
                        continue
                    exact_seen.add(row_hash)
                    parsed_records.append(incoming)
                    continue

                key = _canonical_key(row)
                existing = semantic_seen.get(key)
                if existing is None:
                    semantic_seen[key] = incoming
                    semantic_order.append(key)
                    continue

                resolved = _resolve_record_conflict(
                    existing=existing,
                    incoming=incoming,
                    key=key,
                    conflict_policy=conflict_policy,
                    counts=counts,
                    conflicts=conflicts,
                    stage="semantic_dedupe",
                )
                semantic_seen[key] = resolved

    if dedupe == "semantic_key":
        parsed_records = [semantic_seen[key] for key in semantic_order]

    sampled_model_ids: set[str] = set()
    for row, _, _ in parsed_records:
        if row.get("candidate_source") != "sampled":
            continue
        model_id = str(row.get("sampler_model_id") or "").strip()
        if not model_id:
            continue
        if _parse_int(row.get("sample_index"), default=0) != 0:
            continue
        sampled_model_ids.add(model_id)

    groups: dict[GroupKey, dict[str, Any]] = {}
    group_order: list[GroupKey] = []

    for record in parsed_records:
        row, input_file, line_no = record
        group_key = _group_key(row)
        group = groups.get(group_key)
        if group is None:
            group = {
                "instance_id": row.get("instance_id"),
                "run_id": row.get("run_id"),
                "trajectory_relpath": row.get("trajectory_relpath"),
                "step_index": row.get("step_index"),
                "message_index": row.get("message_index"),
                "problem_id": row.get("problem_id"),
                "gold": None,
                "sampled": {},
            }
            groups[group_key] = group
            group_order.append(group_key)

        source_slot = _row_to_slot_key(row)
        if source_slot is None:
            if row.get("candidate_source") == "sampled" and _parse_int(row.get("sample_index"), default=0) != 0:
                counts["nonzero_sample_rows_ignored"] += 1
            continue

        source_type, model_id, sample_index = source_slot
        if source_type == "gold":
            existing = group["gold"]
            if existing is None:
                group["gold"] = record
                continue
            group["gold"] = _resolve_record_conflict(
                existing=existing,
                incoming=record,
                key=(group_key, source_slot),
                conflict_policy=conflict_policy,
                counts=counts,
                conflicts=conflicts,
                stage="group_slot",
            )
            continue

        sampled_map: dict[str, Record] = group["sampled"]
        existing = sampled_map.get(model_id)
        if existing is None:
            sampled_map[model_id] = record
            continue
        sampled_map[model_id] = _resolve_record_conflict(
            existing=existing,
            incoming=(row, input_file, line_no),
            key=(group_key, (source_type, model_id, sample_index)),
            conflict_policy=conflict_policy,
            counts=counts,
            conflicts=conflicts,
            stage="group_slot",
        )

    counts["groups_total"] = len(group_order)

    merged_rows: list[dict[str, Any]] = []
    dropped_step_keys: set[GroupKey] = set()

    for key in group_order:
        group = groups[key]
        gold_record: Record | None = group["gold"]
        if require_gold and gold_record is None:
            dropped_step_keys.add(key)
            counts["groups_dropped_missing_gold"] += 1
            continue

        candidates_by_source: dict[str, dict[str, Any]] = {}
        actions: list[dict[str, Any]] = []

        if gold_record is not None:
            gold_row = _record_to_public_row(gold_record)
            candidates_by_source["gold"] = gold_row
            gold_action = _first_valid_action(gold_row.get("actions"))
            if gold_action is not None:
                actions.append(
                    {
                        "label": "gold",
                        "candidate_source": "gold",
                        "sampler_model_id": "gold",
                        "sampler_model_name": None,
                        "sample_index": None,
                        "action": gold_action,
                        "has_actions": True,
                    }
                )
                counts["actions_kept_total"] += 1
                counts["actions_kept_gold"] += 1

        sampled_candidates: dict[str, Record] = group["sampled"]
        missing_model_sources: list[str] = []

        for model_id in sorted(sampled_model_ids):
            sampled_record = sampled_candidates.get(model_id)
            if sampled_record is None:
                missing_model_sources.append(model_id)
                continue

            sampled_row = _record_to_public_row(sampled_record)
            candidates_by_source[model_id] = sampled_row
            sampled_action = _first_valid_action(sampled_row.get("actions"))
            if sampled_action is None:
                missing_model_sources.append(model_id)
                continue

            actions.append(
                {
                    "label": model_id,
                    "candidate_source": "sampled",
                    "sampler_model_id": model_id,
                    "sampler_model_name": sampled_row.get("sampler_model_name"),
                    "sample_index": sampled_row.get("sample_index"),
                    "action": sampled_action,
                    "has_actions": True,
                }
            )
            counts["actions_kept_total"] += 1
            counts["actions_kept_sampled"] += 1

        merged_row = {
            "dataset_version": DATASET_VERSION,
            "instance_id": group["instance_id"],
            "run_id": group["run_id"],
            "trajectory_relpath": group["trajectory_relpath"],
            "step_index": group["step_index"],
            "message_index": group["message_index"],
            "problem_id": group["problem_id"],
            "actions": actions,
            "n_actions": len(actions),
            "expected_actions": len(sampled_model_ids) + 1,
            "missing_model_sources": missing_model_sources,
            "candidates_by_source": candidates_by_source,
            "created_at": int(time.time()),
        }
        if len(actions) == 5:
            counts["groups_with_5_actions"] += 1
        merged_rows.append(merged_row)

    counts["missing_gold_steps"] = len(dropped_step_keys)
    counts["missing_gold_rows_dropped"] = len(dropped_step_keys)

    if sort_by_instance_step_model_sample:
        merged_rows.sort(key=_group_sort_key)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")

    per_model_counts: dict[str, int] = {}
    sampled_model_coverage_histogram: dict[str, int] = {}
    for row in merged_rows:
        sampled_models_present: set[str] = set()
        for action_entry in row.get("actions", []):
            if not isinstance(action_entry, dict):
                continue
            model_id = str(action_entry.get("sampler_model_id") or "")
            if not model_id:
                continue
            per_model_counts[model_id] = per_model_counts.get(model_id, 0) + 1
            if action_entry.get("candidate_source") == "sampled":
                sampled_models_present.add(model_id)
        n_models = str(len(sampled_models_present))
        sampled_model_coverage_histogram[n_models] = sampled_model_coverage_histogram.get(n_models, 0) + 1

    summary = {
        "dataset_version": DATASET_VERSION,
        "config": {
            "dedupe": dedupe,
            "conflict_policy": conflict_policy,
            "require_gold": require_gold,
            "sort_by_instance_step_model_sample": sort_by_instance_step_model_sample,
            "sampled_action_selection": "sample_index_0_first_valid_action",
            "missing_sample_policy": "drop_missing_sampled_entries",
            "output_shape": "one_row_per_run_step",
        },
        "inputs": [str(path) for path in input_files],
        "output_jsonl": str(output_jsonl),
        "counts": {
            **counts,
            "rows_kept": len(merged_rows),
            "steps_kept": len({_step_key(row) for row in merged_rows}),
            "groups_kept": len(merged_rows),
        },
        "sampled_model_ids": sorted(sampled_model_ids),
        "per_model_counts": per_model_counts,
        "sampled_model_coverage_histogram": sampled_model_coverage_histogram,
        "dropped_step_keys_missing_gold": [list(key) for key in sorted(dropped_step_keys, key=lambda item: repr(item))],
        "conflict_examples": conflicts[:50],
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary

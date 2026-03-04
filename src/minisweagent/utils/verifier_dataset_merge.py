from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from minisweagent.utils.verifier_action_sampling import DATASET_VERSION

DedupeMode = Literal["none", "exact", "semantic_key"]
ConflictPolicy = Literal["keep_first", "keep_last", "error"]


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
        row.get("message_index"),
        row.get("sampler_model_id"),
        row.get("sample_index"),
        row.get("candidate_source"),
    )


def _step_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("instance_id"), row.get("message_index"))


def _normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _rows_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _normalized_json(a) == _normalized_json(b)


def _sort_key(row: dict[str, Any]) -> tuple[str, int, str, int, str]:
    instance_id = str(row.get("instance_id") or "")
    message_index_raw = row.get("message_index")
    if isinstance(message_index_raw, int):
        message_index = message_index_raw
    else:
        try:
            message_index = int(message_index_raw)
        except (TypeError, ValueError):
            message_index = -1
    sampler_model_id = str(row.get("sampler_model_id") or "")
    sample_index_raw = row.get("sample_index")
    if isinstance(sample_index_raw, int):
        sample_index = sample_index_raw
    else:
        try:
            sample_index = int(sample_index_raw)
        except (TypeError, ValueError):
            sample_index = -1
    source = str(row.get("candidate_source") or "")
    source_rank = "0" if source == "gold" else "1"
    return (instance_id, message_index, source_rank, sample_index, sampler_model_id)


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
    }
    conflicts: list[dict[str, Any]] = []

    merged_rows: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    semantic_seen: dict[tuple[Any, ...], dict[str, Any]] = {}
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

                if dedupe == "none":
                    merged_rows.append(row)
                    continue

                if dedupe == "exact":
                    row_hash = _normalized_json(row)
                    if row_hash in exact_seen:
                        counts["duplicates_dropped"] += 1
                        continue
                    exact_seen.add(row_hash)
                    merged_rows.append(row)
                    continue

                key = _canonical_key(row)
                existing = semantic_seen.get(key)
                if existing is None:
                    semantic_seen[key] = row
                    semantic_order.append(key)
                    continue

                if _rows_equal(existing, row):
                    counts["duplicates_dropped"] += 1
                    continue

                counts["conflicts"] += 1
                conflicts.append(
                    {
                        "input_file": str(input_file),
                        "line": line_no,
                        "key": key,
                    }
                )
                if conflict_policy == "keep_first":
                    continue
                if conflict_policy == "keep_last":
                    semantic_seen[key] = row
                    continue
                if conflict_policy == "error":
                    raise ValueError(
                        f"Conflict for key {key} between existing row and {input_file}:{line_no} "
                        "(set --conflict-policy keep_first|keep_last to resolve automatically)."
                    )

    if dedupe == "semantic_key":
        merged_rows = [semantic_seen[key] for key in semantic_order]

    dropped_step_keys: set[tuple[Any, Any]] = set()
    if require_gold:
        step_has_gold: dict[tuple[Any, Any], bool] = {}
        for row in merged_rows:
            key = _step_key(row)
            if key not in step_has_gold:
                step_has_gold[key] = False
            if row.get("is_gold") is True:
                step_has_gold[key] = True
        dropped_step_keys = {key for key, has_gold in step_has_gold.items() if not has_gold}
        counts["missing_gold_steps"] = len(dropped_step_keys)
        if dropped_step_keys:
            kept: list[dict[str, Any]] = []
            for row in merged_rows:
                if _step_key(row) in dropped_step_keys:
                    counts["missing_gold_rows_dropped"] += 1
                    continue
                kept.append(row)
            merged_rows = kept

    if sort_by_instance_step_model_sample:
        merged_rows.sort(key=_sort_key)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")

    per_model_counts: dict[str, int] = {}
    step_models: dict[tuple[Any, Any], set[str]] = {}
    for row in merged_rows:
        model_id = str(row.get("sampler_model_id") or "")
        per_model_counts[model_id] = per_model_counts.get(model_id, 0) + 1
        if row.get("candidate_source") == "sampled":
            key = _step_key(row)
            if key not in step_models:
                step_models[key] = set()
            step_models[key].add(model_id)
    model_coverage_histogram: dict[str, int] = {}
    for models in step_models.values():
        n_models = str(len(models))
        model_coverage_histogram[n_models] = model_coverage_histogram.get(n_models, 0) + 1

    summary = {
        "dataset_version": DATASET_VERSION,
        "config": {
            "dedupe": dedupe,
            "conflict_policy": conflict_policy,
            "require_gold": require_gold,
            "sort_by_instance_step_model_sample": sort_by_instance_step_model_sample,
        },
        "inputs": [str(path) for path in input_files],
        "output_jsonl": str(output_jsonl),
        "counts": {
            **counts,
            "rows_kept": len(merged_rows),
            "steps_kept": len({_step_key(row) for row in merged_rows}),
        },
        "per_model_counts": per_model_counts,
        "sampled_model_coverage_histogram": model_coverage_histogram,
        "dropped_step_keys_missing_gold": [list(key) for key in sorted(dropped_step_keys, key=lambda item: repr(item))],
        "conflict_examples": conflicts[:50],
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_ALLOWED_OUTPUT_FORMATS = {"list", "rubric_yaml", "auto"}


def _coerce_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _coerce_rubric_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalize_output_payload(payload: Any, *, prompt_name: str | None = None) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = {"items": payload}
    if not isinstance(payload, dict):
        raise ValueError("Precomputed checklist payload must be a mapping or a list of checklist items.")

    items = _coerce_items(payload.get("items"))
    rubric_items = _coerce_rubric_items(payload.get("rubric_items"))
    raw_output = payload.get("raw_output")
    checklist_output_format = str(payload.get("checklist_output_format") or "").strip()
    if checklist_output_format not in _ALLOWED_OUTPUT_FORMATS:
        checklist_output_format = "rubric_yaml" if rubric_items else "list"

    response = payload.get("response")
    if not isinstance(response, dict):
        response = {}

    generator_prompt_name = payload.get("generator_prompt_name")
    if not isinstance(generator_prompt_name, str) or not generator_prompt_name.strip():
        generator_prompt_name = prompt_name or ""

    generator_mode = payload.get("generator_mode")
    if not isinstance(generator_mode, str):
        generator_mode = ""

    return {
        "items": items,
        "rubric_items": rubric_items,
        "checklist_output_format": checklist_output_format,
        "raw_output": str(raw_output or ""),
        "response": response,
        "response_cost": 0.0,
        "api_calls": 0,
        "generator_mode": generator_mode,
        "generator_prompt_name": generator_prompt_name,
    }


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "instance_id" in payload:
            return [payload]
        rows: list[dict[str, Any]] = []
        for instance_id, row_payload in payload.items():
            if not isinstance(instance_id, str) or not instance_id.strip():
                continue
            rows.append({"instance_id": instance_id, "output": row_payload})
        return rows
    raise ValueError("Checklist input must be a JSON object, array, or JSONL file.")


@lru_cache(maxsize=8)
def load_precomputed_checklist_lookup(path_str: str) -> dict[str, dict[str, Any]]:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Precomputed checklist file not found: {path}")

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = _rows_from_payload(json.loads(path.read_text()))

    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            continue
        payload = row.get("output")
        if payload is None:
            payload = row.get("checklist")
        if payload is None:
            payload = row
        if instance_id in lookup:
            raise ValueError(f"Duplicate precomputed checklist for instance_id '{instance_id}' in {path}")
        prompt_name = row.get("prompt_name") if isinstance(row.get("prompt_name"), str) else None
        lookup[instance_id] = _normalize_output_payload(payload, prompt_name=prompt_name)
    return lookup


def resolve_precomputed_checklist(
    config: Any,
    *,
    instance_id: str | None,
) -> dict[str, Any] | None:
    path = getattr(config, "checklist_input_path", None)
    if not isinstance(path, str) or not path.strip():
        return None
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("Precomputed checklist input requires an instance_id.")

    lookup = load_precomputed_checklist_lookup(path)
    resolved = lookup.get(instance_id)
    if resolved is None:
        behavior = str(getattr(config, "checklist_input_missing_behavior", "error") or "error").strip()
        if behavior == "generate":
            return None
        raise KeyError(f"No precomputed checklist found for instance_id '{instance_id}' in {path}")

    payload = copy.deepcopy(resolved)
    payload.setdefault("source", "precomputed_checklist")
    return payload

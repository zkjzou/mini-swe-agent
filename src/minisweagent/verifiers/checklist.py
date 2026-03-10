from __future__ import annotations

import re
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template

from minisweagent.verifiers.query_utils import (
    build_verifier_messages,
    query_verifier_text,
    resolve_verifier_history_message_format,
)

_DEFAULT_CHECKLIST_ITEMS = [
    "Reproduce and confirm the issue behavior.",
    "Locate the root cause in the source code.",
    "Implement a minimal, targeted fix in non-test files.",
    "Run focused validation to confirm the fix.",
    "Check for regressions and ensure task requirements are met.",
]


def resolve_checklist_output_format(config: Any) -> str:
    """Resolve checklist output parsing mode from explicit config or prompt variant."""
    configured_mode = str(getattr(config, "checklist_output_format", "auto") or "auto").strip().lower()
    if configured_mode in {"list", "rubric_yaml"}:
        return configured_mode

    prompt_name = getattr(config, "prompt_name", None)
    if isinstance(prompt_name, str) and prompt_name.startswith("checklist_v2/"):
        return "rubric_yaml"
    return "list"


def infer_checklist_prompt_settings(prompt_name: str | None) -> dict[str, Any] | None:
    if not isinstance(prompt_name, str):
        return None
    name = prompt_name.strip()
    if not name:
        return None
    if name.startswith("dynamic_checklist_modify/"):
        return {
            "checklist_mode": "issue_progress",
            "checklist_dynamic": True,
            "checklist_update_mode": "modify",
        }
    if name.startswith("dynamic_checklist_regenerate/"):
        return {
            "checklist_mode": "issue_progress",
            "checklist_dynamic": True,
            "checklist_update_mode": "regenerate",
        }
    if name.startswith("checklist/") or name.startswith("checklist_v2/"):
        return {
            "checklist_mode": "issue_progress",
            "checklist_dynamic": False,
            "checklist_update_mode": "regenerate",
        }
    return None


def generate_issue_checklist(
    model: Any,
    config: Any,
    *,
    template_vars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a reusable issue-progress checklist from the task description."""
    template_vars = dict(template_vars or {})
    template_vars.setdefault("history_message_format", resolve_verifier_history_message_format(config))
    system_prompt = _render(getattr(config, "checklist_system_template"), **template_vars)
    checklist_prompt = _render(getattr(config, "checklist_prompt_template"), **template_vars)
    verifier_messages = build_verifier_messages(
        config=config,
        system_prompt=system_prompt,
        final_user_prompt=checklist_prompt,
        template_vars=template_vars,
    )
    content, response, response_cost = query_verifier_text(
        model,
        verifier_messages,
    )
    output_format = resolve_checklist_output_format(config)
    rubric_items = parse_checklist_rubric(content)

    if rubric_items:
        items = dedupe_checklist_items([item["description"] for item in rubric_items])
    else:
        items = parse_checklist_items(content, item_regex=getattr(config, "checklist_item_regex"))

    if output_format == "rubric_yaml":
        items = dedupe_checklist_items(items)
    else:
        items = normalize_checklist_items(
            items,
            min_items=getattr(config, "checklist_min_items"),
            max_items=getattr(config, "checklist_max_items"),
        )
    output = {
        "items": items,
        "rubric_items": rubric_items,
        "checklist_output_format": output_format,
        "raw_output": content,
        "response": response,
        "response_cost": response_cost,
        "api_calls": 1,
    }
    if getattr(config, "include_inputs_in_output", False):
        output["input"] = {"messages": verifier_messages}
    return output


def parse_checklist_items(content: str, *, item_regex: str) -> list[str]:
    """Parse checklist entries from model output."""
    items: list[str] = []
    for item in _parse_by_regex(content, item_regex):
        cleaned = _clean_item(item)
        if cleaned:
            items.append(cleaned)
    if items:
        return items
    for item in _parse_bullets(content):
        cleaned = _clean_item(item)
        if cleaned:
            items.append(cleaned)
    return items


def normalize_checklist_items(items: list[str], *, min_items: int, max_items: int) -> list[str]:
    """Deduplicate and clamp checklist entries, then fill missing entries if needed."""
    max_items = max(1, int(max_items))
    min_items = max(0, min(int(min_items), max_items))
    deduped = dedupe_checklist_items(items, max_items=max_items)
    seen = {_normalize_key(item) for item in deduped}

    if len(deduped) < min_items:
        for fallback_item in _DEFAULT_CHECKLIST_ITEMS:
            key = _normalize_key(fallback_item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fallback_item)
            if len(deduped) >= min_items:
                break

    return deduped[:max_items]


def dedupe_checklist_items(items: list[str], *, max_items: int | None = None) -> list[str]:
    """Deduplicate checklist entries while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def parse_checklist_rubric(content: str) -> list[dict[str, Any]]:
    """Parse YAML rubric checklist output into normalized structured checklist items."""
    try:
        parsed = yaml.safe_load(content)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []

    rubric = parsed.get("rubric")
    if not isinstance(rubric, list):
        return []

    normalized_items: list[dict[str, Any]] = []
    for idx, entry in enumerate(rubric):
        if not isinstance(entry, dict):
            continue
        description = _clean_optional_text(entry.get("description"))
        if not description:
            continue
        normalized: dict[str, Any] = {
            "id": _clean_optional_text(entry.get("id")) or f"S{idx + 1}",
            "description": description,
        }
        stage = _clean_optional_text(entry.get("stage")) or _clean_optional_text(entry.get("phase"))
        if stage:
            normalized["stage"] = stage
        observable_signal = _clean_optional_text(entry.get("observable_signal")) or _clean_optional_text(
            entry.get("done_when")
        )
        if observable_signal:
            normalized["observable_signal"] = observable_signal
        weight = _coerce_weight(entry.get("weight"))
        if weight is not None:
            normalized["weight"] = weight
        normalized_items.append(normalized)
    return normalized_items


def _render(template: str, **kwargs) -> str:
    return Template(template, undefined=StrictUndefined).render(**kwargs)


def _parse_by_regex(content: str, item_regex: str) -> list[str]:
    parsed: list[str] = []
    for match in re.finditer(item_regex, content, re.MULTILINE):
        if match.groups():
            parsed.append(match.group(match.lastindex or 1))
        else:
            parsed.append(match.group(0))
    return parsed


def _parse_bullets(content: str) -> list[str]:
    parsed: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            continue
        match = re.match(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$", line)
        if match:
            parsed.append(match.group(1))
    return parsed


def _clean_item(item: str) -> str:
    cleaned = item.strip()
    cleaned = re.sub(r"^\[[ xX]\]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _normalize_key(item: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
    tokens = [token for token in lowered.split() if token not in {"a", "an", "the", "to", "of", "for", "and"}]
    return " ".join(tokens)


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return re.sub(r"\s+", " ", cleaned)


def _coerce_weight(value: Any) -> int | None:
    try:
        weight = int(value)
    except (TypeError, ValueError):
        return None
    return weight if 1 <= weight <= 3 else None

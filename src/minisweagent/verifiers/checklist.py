from __future__ import annotations

import re
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent.verifiers.query_utils import query_verifier_text

_DEFAULT_CHECKLIST_ITEMS = [
    "Reproduce and confirm the issue behavior.",
    "Locate the root cause in the source code.",
    "Implement a minimal, targeted fix in non-test files.",
    "Run focused validation to confirm the fix.",
    "Check for regressions and ensure task requirements are met.",
]


def generate_issue_checklist(
    model: Any,
    config: Any,
    *,
    template_vars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a reusable issue-progress checklist from the task description."""
    template_vars = template_vars or {}
    system_prompt = _render(getattr(config, "checklist_system_template"), **template_vars)
    checklist_prompt = _render(getattr(config, "checklist_prompt_template"), **template_vars)
    content, response, response_cost = query_verifier_text(
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": checklist_prompt},
        ],
    )
    items = parse_checklist_items(content, item_regex=getattr(config, "checklist_item_regex"))
    items = normalize_checklist_items(
        items,
        min_items=getattr(config, "checklist_min_items"),
        max_items=getattr(config, "checklist_max_items"),
    )
    return {
        "items": items,
        "raw_output": content,
        "response": response,
        "response_cost": response_cost,
    }


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
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_items:
            break

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

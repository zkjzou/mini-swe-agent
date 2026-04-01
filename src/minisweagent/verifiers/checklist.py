from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template

from minisweagent import package_dir
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
_GENERATOR_ITEM_DEFAULTS = [
    ("D1", "diagnose", 3, "Confirm the current failure or missing behavior from the existing context."),
    ("D2", "localize", 3, "Identify the relevant source area to inspect or update."),
    ("D3", "fix", 3, "Apply a targeted implementation change that addresses the issue."),
    ("D4", "validate", 2, "Run focused validation for the affected behavior."),
    ("D5", "cleanup", 2, "Check for regressions and avoid unrelated churn."),
]
_DEFAULT_GENERATOR_PROMPTS = {
    "trajectory_success": "static_success",
    "trajectory_failure": "static_failure",
    "trajectory_pairwise": "pairwise_evolve",
    "trajectory_dynamic": "dynamic_success",
}


def resolve_checklist_output_format(config: Any) -> str:
    """Resolve checklist output parsing mode from explicit config or prompt variant."""
    configured_mode = str(getattr(config, "checklist_output_format", "auto") or "auto").strip().lower()
    if configured_mode in {"list", "rubric_yaml"}:
        return configured_mode

    prompt_name = getattr(config, "prompt_name", None)
    if isinstance(prompt_name, str) and (
        prompt_name.startswith("checklist_v2/")
        or prompt_name.startswith("ultimate_v2/")
        or prompt_name.startswith("ultimate_v2_mini/")
        or prompt_name.startswith("ultimate_v2_dynamic_checklist_regenerate/")
    ):
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
    if name.startswith("ultimate_v2_dynamic_checklist_regenerate/"):
        return {
            "checklist_mode": "issue_progress",
            "checklist_dynamic": True,
            "checklist_update_mode": "regenerate",
        }
    if (
        name.startswith("checklist/")
        or name.startswith("checklist_v2/")
        or name.startswith("ultimate_v2/")
        or name.startswith("ultimate_v2_mini/")
    ):
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
    template_vars = _prepare_checklist_generator_template_vars(config, template_vars)
    template_vars.setdefault("history_message_format", resolve_verifier_history_message_format(config))
    generation_config = _resolve_checklist_generation_config(config)
    system_prompt = _render(getattr(generation_config, "checklist_system_template"), **template_vars)
    checklist_prompt = _render(getattr(generation_config, "checklist_prompt_template"), **template_vars)
    verifier_messages = build_verifier_messages(
        config=generation_config,
        system_prompt=system_prompt,
        final_user_prompt=checklist_prompt,
        template_vars=template_vars,
    )
    content, response, response_cost = query_verifier_text(
        model,
        verifier_messages,
    )
    rubric_items = parse_checklist_rubric(content)
    output_format = "rubric_yaml" if rubric_items else resolve_checklist_output_format(config)

    if rubric_items:
        items = dedupe_checklist_items([item["description"] for item in rubric_items])
    else:
        items = parse_checklist_items(
            content,
            item_regex=getattr(config, "checklist_item_regex", r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$"),
        )

    if output_format == "rubric_yaml":
        items = dedupe_checklist_items(items)
    else:
        items = normalize_checklist_items(
            items,
            min_items=getattr(config, "checklist_min_items", 3),
            max_items=getattr(config, "checklist_max_items", 8),
        )
    sanitized_output = sanitize_checklist_generation_output(
        config,
        template_vars=template_vars,
        items=items,
        rubric_items=rubric_items,
    )
    output = {
        "items": sanitized_output["items"],
        "rubric_items": sanitized_output["rubric_items"],
        "checklist_output_format": output_format,
        "raw_output": content,
        "response": response,
        "response_cost": response_cost,
        "api_calls": 1,
        "generator_mode": getattr(generation_config, "checklist_generator_mode", "issue_only"),
        "generator_prompt_name": resolve_checklist_generator_prompt_name(generation_config),
    }
    if sanitized_output["guardrail"]:
        output["guardrail"] = sanitized_output["guardrail"]
    if getattr(config, "include_inputs_in_output", False):
        output["input"] = {"messages": verifier_messages}
    return output


def load_checklist_generator_templates(config: Any) -> tuple[str | None, str | None]:
    prompt_name = resolve_checklist_generator_prompt_name(config)
    if not isinstance(prompt_name, str) or not prompt_name.strip():
        return None, None

    prompt_root = _resolve_prompt_root(
        Path(getattr(config, "checklist_generator_prompt_dir", "prompts/checklist_generator")),
        prompt_name.strip(),
    )
    system_path = prompt_root / "system.jinja"
    prompt_path = prompt_root / "prompt.jinja"
    system_template = system_path.read_text() if system_path.is_file() else None
    prompt_template = prompt_path.read_text() if prompt_path.is_file() else None
    return system_template, prompt_template


def resolve_checklist_generator_prompt_name(config: Any) -> str | None:
    prompt_name = getattr(config, "checklist_generator_prompt_name", None)
    if isinstance(prompt_name, str) and prompt_name.strip():
        return prompt_name.strip()
    generator_mode = str(getattr(config, "checklist_generator_mode", "issue_only") or "issue_only").strip()
    return _DEFAULT_GENERATOR_PROMPTS.get(generator_mode)


def sanitize_checklist_generation_output(
    config: Any,
    *,
    template_vars: dict[str, Any],
    items: list[str],
    rubric_items: list[dict[str, Any]],
) -> dict[str, Any]:
    if getattr(config, "checklist_generator_mode", "issue_only") != "trajectory_dynamic":
        return {"items": items, "rubric_items": rubric_items, "guardrail": None}
    if not getattr(config, "checklist_generator_validate_grounding", True):
        return {"items": items, "rubric_items": rubric_items, "guardrail": None}

    prior_text = _collect_text(template_vars.get("messages")) + "\n" + _collect_text(template_vars.get("task"))
    future_text = _collect_text(template_vars.get("future_steps")) + "\n" + _collect_text(
        template_vars.get("future_steps_text")
    )
    future_only_terms = _extract_grounding_terms(future_text) - _extract_grounding_terms(prior_text)
    if not future_only_terms:
        return {"items": items, "rubric_items": rubric_items, "guardrail": None}

    sanitized_items = [_sanitize_future_only_details(item, future_only_terms) for item in items]
    sanitized_items = dedupe_checklist_items([item for item in sanitized_items if item])

    sanitized_rubric_items: list[dict[str, Any]] = []
    for item in rubric_items:
        sanitized = dict(item)
        sanitized["description"] = _sanitize_future_only_details(sanitized.get("description", ""), future_only_terms)
        observable_signal = sanitized.get("observable_signal")
        if isinstance(observable_signal, str):
            sanitized["observable_signal"] = _sanitize_future_only_details(observable_signal, future_only_terms)
        if sanitized.get("description"):
            sanitized_rubric_items.append(sanitized)

    if sanitized_rubric_items and not sanitized_items:
        sanitized_items = [item["description"] for item in sanitized_rubric_items]
    if not sanitized_rubric_items and getattr(config, "checklist_output_format", "auto") == "rubric_yaml":
        sanitized_rubric_items = _fallback_rubric_items(sanitized_items)
    if not sanitized_items:
        sanitized_items = normalize_checklist_items(
            [],
            min_items=getattr(config, "checklist_min_items", 3),
            max_items=getattr(config, "checklist_max_items", 8),
        )
        if getattr(config, "checklist_output_format", "auto") == "rubric_yaml":
            sanitized_rubric_items = _fallback_rubric_items(sanitized_items)

    return {
        "items": sanitized_items,
        "rubric_items": sanitized_rubric_items,
        "guardrail": {
            "mode": "trajectory_dynamic",
            "sanitized_terms": sorted(future_only_terms),
            "sanitized_term_count": len(future_only_terms),
        },
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


def _resolve_checklist_generation_config(config: Any) -> Any:
    system_template, prompt_template = load_checklist_generator_templates(config)
    if system_template is None and prompt_template is None:
        return config

    resolved = getattr(config, "model_copy", None)
    if callable(resolved):
        updated = config.model_copy(deep=True)
    else:
        updated = _ChecklistConfigProxy(config)
    if system_template is not None:
        updated.checklist_system_template = system_template
    if prompt_template is not None:
        updated.checklist_prompt_template = prompt_template
    return updated


def _prepare_checklist_generator_template_vars(config: Any, template_vars: dict[str, Any]) -> dict[str, Any]:
    generator_mode = str(getattr(config, "checklist_generator_mode", "issue_only") or "issue_only").strip()
    if generator_mode == "issue_only" and not resolve_checklist_generator_prompt_name(config):
        return template_vars
    from minisweagent.verifiers.checklist_generator import prepare_checklist_generator_template_vars

    return prepare_checklist_generator_template_vars(
        template_vars,
        generator_mode=generator_mode or "issue_only",
    )


def _resolve_prompt_root(prompt_dir: Path, prompt_name: str) -> Path:
    if prompt_dir.is_absolute():
        return prompt_dir / prompt_name
    cwd_candidate = Path.cwd() / prompt_dir / prompt_name
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root_candidate = package_dir.parent.parent / prompt_dir / prompt_name
    if repo_root_candidate.exists():
        return repo_root_candidate
    return cwd_candidate


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


def _collect_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_collect_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_collect_text(item) for item in value)
    return ""


def _extract_grounding_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.findall(r"`([^`]+)`", text):
        normalized = match.strip().lower()
        if normalized:
            terms.add(normalized)
    for match in re.findall(r"\b[a-zA-Z0-9_./-]{4,}\b", text):
        normalized = match.strip().lower()
        if any(ch in normalized for ch in "/._-") or re.search(r"[a-z][A-Z]|[A-Z]{2,}", match):
            terms.add(normalized)
    return terms


def _sanitize_future_only_details(text: str, future_only_terms: set[str]) -> str:
    sanitized = text
    for term in sorted(future_only_terms, key=len, reverse=True):
        replacement = _replacement_for_term(term)
        sanitized = re.sub(re.escape(term), replacement, sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def _replacement_for_term(term: str) -> str:
    if "/" in term or "." in term:
        return "relevant implementation detail"
    if re.fullmatch(r"[a-z0-9_]+", term):
        return "relevant code element"
    return "relevant code element"


def _fallback_rubric_items(items: list[str]) -> list[dict[str, Any]]:
    rubric_items: list[dict[str, Any]] = []
    source_items = items or [item[3] for item in _GENERATOR_ITEM_DEFAULTS]
    for index, description in enumerate(source_items, start=1):
        default_id, default_phase, default_weight, default_description = _GENERATOR_ITEM_DEFAULTS[
            min(index - 1, len(_GENERATOR_ITEM_DEFAULTS) - 1)
        ]
        rubric_items.append(
            {
                "id": f"S{index}" if index > len(_GENERATOR_ITEM_DEFAULTS) else default_id,
                "stage": default_phase,
                "weight": default_weight,
                "description": description or default_description,
                "observable_signal": "Progress is visible from the prior trajectory context.",
            }
        )
    return rubric_items


class _ChecklistConfigProxy:
    def __init__(self, source: Any):
        self.__dict__.update(getattr(source, "__dict__", {}))
        self._source = source

    def __getattr__(self, item: str) -> Any:
        return getattr(self._source, item)

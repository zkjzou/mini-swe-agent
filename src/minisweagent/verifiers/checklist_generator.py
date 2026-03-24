from __future__ import annotations

import copy
import re
from typing import Any

from minisweagent.verifiers.checklist import (
    generate_issue_checklist,
    load_checklist_generator_templates,
    parse_checklist_rubric,
    resolve_checklist_generator_prompt_name,
)


_TEXT_ONLY_MODEL_CLASS_DEFAULT = "litellm_textbased"
_TEXT_ONLY_MODEL_CLASS_REWRITES = {
    "litellm": "litellm_textbased",
    "minisweagent.models.litellm_model.LitellmModel": "litellm_textbased",
    "openrouter": "openrouter_textbased",
    "minisweagent.models.openrouter_model.OpenRouterModel": "openrouter_textbased",
}
_TEXT_ONLY_MODEL_CLASS_ALLOWED = {
    "litellm_textbased",
    "openrouter_textbased",
    "deterministic",
    "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel",
    "minisweagent.models.openrouter_textbased_model.OpenRouterTextbasedModel",
    "minisweagent.models.test_models.DeterministicModel",
}


def normalize_checklist_generator_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(model_config)
    requested_model_class = str(normalized.get("model_class") or "").strip()
    if not requested_model_class:
        normalized["model_class"] = _TEXT_ONLY_MODEL_CLASS_DEFAULT
        return normalized
    if rewritten_class := _TEXT_ONLY_MODEL_CLASS_REWRITES.get(requested_model_class):
        normalized["model_class"] = rewritten_class
        return normalized
    if requested_model_class in _TEXT_ONLY_MODEL_CLASS_ALLOWED:
        return normalized
    allowed = ", ".join(sorted(_TEXT_ONLY_MODEL_CLASS_ALLOWED))
    raise ValueError(
        f"Unsupported checklist generator model_class '{requested_model_class}'. "
        f"Checklist generation must use a text-based verifier model. Use one of: {allowed}"
    )


def resolve_checklist_generator_model_config(config: Any) -> dict[str, Any]:
    configured = getattr(config, "checklist_generator_model", None)
    if isinstance(configured, dict) and configured:
        return normalize_checklist_generator_model_config(configured)

    fallback = getattr(config, "model", None)
    if isinstance(fallback, dict):
        return normalize_checklist_generator_model_config(fallback)
    return normalize_checklist_generator_model_config({})


def prepare_checklist_generator_template_vars(
    template_vars: dict[str, Any],
    *,
    generator_mode: str,
) -> dict[str, Any]:
    prepared = dict(template_vars)
    prepared["generator_mode"] = generator_mode
    visible_messages = _coerce_messages(prepared.get("messages"))
    visible_steps = _coerce_steps(prepared.get("steps"))
    all_steps = _coerce_steps(prepared.get("all_steps"))

    if not visible_messages and visible_steps:
        visible_messages = [message for step in visible_steps for message in step]

    all_messages = _coerce_messages(prepared.get("trajectory_messages"))
    if not all_messages and all_steps:
        all_messages = [message for step in all_steps for message in step]
    if not all_messages:
        all_messages = list(visible_messages)

    full_trajectory_text = _render_messages_text(all_messages or visible_messages)
    prior_trajectory_text = _render_messages_text(visible_messages)
    prepared.setdefault("trajectory_messages", all_messages)
    prepared.setdefault("full_trajectory_text", full_trajectory_text)
    prepared.setdefault("prior_trajectory_text", prior_trajectory_text)
    prepared.setdefault("current_trajectory", prior_trajectory_text)
    prepared.setdefault("successful_trajectory_text", full_trajectory_text)
    prepared.setdefault("unsuccessful_trajectory_text", full_trajectory_text)
    prepared.setdefault("remaining_successful_trajectory_text", "")
    prepared.setdefault("remaining_unsuccessful_trajectory_text", "")

    if generator_mode == "trajectory_dynamic":
        future_messages = all_messages[len(visible_messages) :]
        future_trajectory_text = _render_messages_text(future_messages)
        prepared["remaining_successful_trajectory_text"] = future_trajectory_text
        prepared["remaining_unsuccessful_trajectory_text"] = future_trajectory_text
        future_steps = prepared.get("future_steps")
        if not isinstance(future_steps, list) or not future_steps:
            future_steps = [_summarize_message(message) for message in future_messages]
        prepared["messages"] = visible_messages
        prepared["trajectory_text"] = prior_trajectory_text
        prepared["future_steps"] = future_steps
        prepared["future_steps_text"] = _render_future_steps_text(future_steps)
        return prepared

    prepared["messages"] = visible_messages or all_messages
    prepared.setdefault("trajectory_text", full_trajectory_text)
    prepared.setdefault("future_steps", [])
    prepared.setdefault("future_steps_text", "")
    return prepared


def generate_trajectory_checklist(
    model: Any,
    config: Any,
    *,
    prompt_name: str,
    template_vars: dict[str, Any] | None = None,
    prompt_dir: str | None = None,
) -> dict[str, Any]:
    updated = getattr(config, "model_copy", None)
    resolved = config.model_copy(deep=True) if callable(updated) else _ChecklistGeneratorConfigProxy(config)
    resolved_mode = (
        (template_vars or {}).get("generator_mode")
        or getattr(config, "checklist_generator_mode", None)
        or _generator_mode_from_prompt_name(prompt_name)
        or "trajectory_success"
    )
    prepared_vars = prepare_checklist_generator_template_vars(
        dict(template_vars or {}),
        generator_mode=str(resolved_mode),
    )
    resolved.checklist_generator_mode = prepared_vars.get("generator_mode", resolved_mode)
    resolved.checklist_generator_prompt_name = prompt_name
    if isinstance(getattr(resolved, "model", None), dict):
        resolved.model = normalize_checklist_generator_model_config(resolved.model)
    if isinstance(getattr(resolved, "checklist_generator_model", None), dict):
        resolved.checklist_generator_model = normalize_checklist_generator_model_config(resolved.checklist_generator_model)
    if prompt_dir is not None:
        resolved.checklist_generator_prompt_dir = prompt_dir
    return generate_issue_checklist(model, resolved, template_vars=prepared_vars)


class _ChecklistGeneratorConfigProxy:
    def __init__(self, config: Any):
        self._config = config

    def __getattr__(self, item: str) -> Any:
        return getattr(self._config, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_config":
            object.__setattr__(self, key, value)
            return
        object.__setattr__(self, key, value)


__all__ = [
    "generate_trajectory_checklist",
    "load_checklist_generator_templates",
    "resolve_checklist_generator_model_config",
    "normalize_checklist_generator_model_config",
    "prepare_checklist_generator_template_vars",
    "resolve_checklist_generator_prompt_name",
]


def _coerce_steps(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    return [[message for message in step if isinstance(message, dict)] for step in value if isinstance(step, list)]


def _coerce_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, dict)]


def _summarize_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    content = str(message.get("content") or "").strip()
    content = " ".join(content.split())
    return f"{role}: {content}" if content else role


def _render_messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_summarize_message(message) for message in messages)


def _render_future_steps_text(steps: list[Any]) -> str:
    if not steps:
        return "Full future steps after the current step:\n<none>"
    rendered = ["Full future steps after the current step:"]
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            text = _summarize_message(step)
        else:
            text = str(step).strip()
        rendered.append(f"{index}. {text}")
    return "\n".join(rendered)


def _generator_mode_from_prompt_name(prompt_name: str) -> str | None:
    return {
        "static_success": "trajectory_success",
        "static_success_v2": "trajectory_success",
        "static_failure": "trajectory_failure",
        "static_failure_v2": "trajectory_failure",
        "pairwise_evolve": "trajectory_pairwise",
        "dynamic_success": "trajectory_dynamic",
        "dynamic_success_v2": "trajectory_dynamic",
        "dynamic_success_minimal": "trajectory_dynamic",
        "dynamic_failure": "trajectory_dynamic",
        "dynamic_failure_v2": "trajectory_dynamic",
        "dynamic_failure_minimal": "trajectory_dynamic",
        "static_success_minimal": "trajectory_success",
        "static_failure_minimal": "trajectory_failure",
    }.get(prompt_name)

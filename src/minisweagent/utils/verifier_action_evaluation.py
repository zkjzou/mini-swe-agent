from __future__ import annotations

import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from minisweagent.agents.default import (
    VerifierConfig,
    _assert_safe_verifier_fallback_model,
    _normalize_verifier_model_config,
)
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.run.benchmarks.swebench import DEFAULT_CONFIG_FILE, _resolve_profiled_model_config
from minisweagent.utils.langfuse import (
    attach_langfuse_session_metadata,
    enable_langfuse_tracing,
    make_langfuse_session_id,
)
from minisweagent.utils.serialize import recursive_merge
from minisweagent.verifiers.checklist import (
    generate_issue_checklist,
    resolve_checklist_output_format,
)
from minisweagent.verifiers.first_valid import FirstValidVerifier
from minisweagent.verifiers.llm import LLMVerifier
from minisweagent.verifiers.reward_model import RewardModelVerifier
from minisweagent.verifiers.runtime_config import resolve_verifier_runtime_config, verifier_uses_checklist_mode

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm may be unavailable in some environments
    _tqdm = None

VerifierType = Literal["first_valid", "llm", "reward_model"]


@dataclass(frozen=True)
class _VerifierVariantSpec:
    name: str
    verifier_type: VerifierType
    prompt_name: str | None = None
    prompt_dir: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)


_WORLD_SELECTION_SCORE_REGEX = (
    r"(?ms)Candidate\s+(\d+)\s*:\s*.*?SCORES:\s*([+-]?\d+(?:\.\d+)?)"
)
_FINAL_SELECTION_REGEX = r"FINAL:\s*(\d+)"
_VERIFIER_SCORE_REGEX = r"(?ms)Candidate\s+(\d+)\s*:\s*.*?SCORE:\s*([+-]?\d+(?:\.\d+)?)"
_CHECKLIST_STATIC_OVERRIDES = {
    "checklist_mode": "issue_progress",
    "checklist_dynamic": False,
    "checklist_update_mode": "regenerate",
}
_DYNAMIC_CHECKLIST_REGENERATE_OVERRIDES = {
    "checklist_mode": "issue_progress",
    "checklist_dynamic": True,
    "checklist_update_mode": "regenerate",
}
_DYNAMIC_CHECKLIST_MODIFY_OVERRIDES = {
    "checklist_mode": "issue_progress",
    "checklist_dynamic": True,
    "checklist_update_mode": "modify",
}
_NON_CHECKLIST_OVERRIDES = {
    "checklist_mode": "off",
    "checklist_dynamic": False,
    "checklist_update_mode": "regenerate",
}
_LLM_PROMPT_OVERRIDES = {
    "selection_regex": _FINAL_SELECTION_REGEX,
    "selection_score_regex": _VERIFIER_SCORE_REGEX,
}
_VERIFIER_VARIANTS: tuple[_VerifierVariantSpec, ...] = (
    _VerifierVariantSpec(name="first_valid", verifier_type="first_valid"),
    _VerifierVariantSpec(
        name="basic_verifier",
        verifier_type="llm",
        prompt_name="basic/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_NON_CHECKLIST_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="basic_mini_verifier",
        verifier_type="llm",
        prompt_name="basic_mini/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_NON_CHECKLIST_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="domain_verifier",
        verifier_type="llm",
        prompt_name="domain/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_NON_CHECKLIST_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="domain_v2_verifier",
        verifier_type="llm",
        prompt_name="domain_v2/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_NON_CHECKLIST_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="world_verifier",
        verifier_type="llm",
        prompt_name="world/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={
            **_NON_CHECKLIST_OVERRIDES,
            "selection_regex": _FINAL_SELECTION_REGEX,
            "selection_score_regex": _WORLD_SELECTION_SCORE_REGEX,
        },
    ),
    _VerifierVariantSpec(
        name="checklist_verifier",
        verifier_type="llm",
        prompt_name="checklist/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_CHECKLIST_STATIC_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="checklist_v2_verifier",
        verifier_type="llm",
        prompt_name="checklist_v2/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_CHECKLIST_STATIC_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="dynamic_checklist_regenerate_verifier",
        verifier_type="llm",
        prompt_name="dynamic_checklist_regenerate/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_DYNAMIC_CHECKLIST_REGENERATE_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="dynamic_checklist_modify_verifier",
        verifier_type="llm",
        prompt_name="dynamic_checklist_modify/verifier",
        prompt_dir="prompts/verifier",
        config_overrides={**_DYNAMIC_CHECKLIST_MODIFY_OVERRIDES, **_LLM_PROMPT_OVERRIDES},
    ),
    _VerifierVariantSpec(
        name="basic_reward",
        verifier_type="reward_model",
        prompt_name="basic/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_NON_CHECKLIST_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="basic_mini_reward",
        verifier_type="reward_model",
        prompt_name="basic_mini/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_NON_CHECKLIST_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="domain_reward",
        verifier_type="reward_model",
        prompt_name="domain/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_NON_CHECKLIST_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="domain_v2_reward",
        verifier_type="reward_model",
        prompt_name="domain_v2/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_NON_CHECKLIST_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="world_reward",
        verifier_type="reward_model",
        prompt_name="world/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_NON_CHECKLIST_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="checklist_reward",
        verifier_type="reward_model",
        prompt_name="checklist/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_CHECKLIST_STATIC_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="checklist_v2_reward",
        verifier_type="reward_model",
        prompt_name="checklist_v2/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_CHECKLIST_STATIC_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="dynamic_checklist_regenerate_reward",
        verifier_type="reward_model",
        prompt_name="dynamic_checklist_regenerate/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_DYNAMIC_CHECKLIST_REGENERATE_OVERRIDES,
    ),
    _VerifierVariantSpec(
        name="dynamic_checklist_modify_reward",
        verifier_type="reward_model",
        prompt_name="dynamic_checklist_modify/reward",
        prompt_dir="prompts/verifier",
        config_overrides=_DYNAMIC_CHECKLIST_MODIFY_OVERRIDES,
    ),
)
_VERIFIER_VARIANT_BY_NAME = {spec.name: spec for spec in _VERIFIER_VARIANTS}


@dataclass
class _VerifierSession:
    variant_name: str
    verifier_type: VerifierType
    config: VerifierConfig
    verifier: FirstValidVerifier | LLMVerifier | RewardModelVerifier
    checklist_cache: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)


def _normalize_verifier_types(verifier_types: list[str] | None) -> list[VerifierType]:
    if not verifier_types:
        return ["first_valid", "llm", "reward_model"]
    resolved: list[VerifierType] = []
    for verifier_type in verifier_types:
        raw = str(verifier_type or "").strip()
        if raw not in {"first_valid", "llm", "reward_model"}:
            raise ValueError(
                f"Unsupported verifier type '{verifier_type}'. Expected first_valid, llm, or reward_model."
            )
        typed = raw  # type: ignore[assignment]
        if typed not in resolved:
            resolved.append(typed)
    if not resolved:
        raise ValueError("No verifier types selected.")
    return resolved


def _load_resolved_config(config_specs: list[str] | None) -> tuple[dict[str, Any], list[str]]:
    specs = [str(spec) for spec in (config_specs or []) if str(spec).strip()]
    if not specs:
        specs = [str(DEFAULT_CONFIG_FILE)]
    configs = [get_config_from_spec(spec) for spec in specs]
    merged = recursive_merge(*configs)
    resolved = _resolve_profiled_model_config(merged)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved config must be a mapping.")
    return resolved, specs


def _default_prompt_name(verifier_type: VerifierType) -> str:
    if verifier_type == "llm":
        return "basic/verifier"
    if verifier_type == "reward_model":
        return "basic/reward"
    return ""


def _align_prompt_name(prompt_name: str | None, verifier_type: VerifierType) -> str:
    if verifier_type == "first_valid":
        return ""
    if not isinstance(prompt_name, str) or not prompt_name.strip():
        return _default_prompt_name(verifier_type)
    name = prompt_name.strip()
    if verifier_type == "reward_model" and name.endswith("/verifier"):
        return f"{name.rsplit('/', 1)[0]}/reward"
    if verifier_type == "llm" and name.endswith("/reward"):
        return f"{name.rsplit('/', 1)[0]}/verifier"
    return name


def _resolve_verifier_variants(
    verifier_types: list[str] | None,
    verifier_variants: list[str] | None,
) -> list[_VerifierVariantSpec]:
    if verifier_variants:
        resolved_variants: list[_VerifierVariantSpec] = []
        for verifier_variant in verifier_variants:
            name = str(verifier_variant or "").strip()
            spec = _VERIFIER_VARIANT_BY_NAME.get(name)
            if spec is None:
                available = ", ".join(sorted(_VERIFIER_VARIANT_BY_NAME)) or "<none>"
                raise ValueError(f"Unsupported verifier variant '{verifier_variant}'. Available variants: {available}")
            if spec not in resolved_variants:
                resolved_variants.append(spec)
        if not resolved_variants:
            raise ValueError("No verifier variants selected.")
        return resolved_variants

    resolved_types = _normalize_verifier_types(verifier_types)
    return [spec for spec in _VERIFIER_VARIANTS if spec.verifier_type in resolved_types]


def _requires_serial_evaluation(config: dict[str, Any], variant_specs: list[_VerifierVariantSpec]) -> bool:
    return any(verifier_uses_checklist_mode(_resolve_variant_verifier_config(config, spec)) for spec in variant_specs)


def _resolve_variant_verifier_config(config: dict[str, Any], variant_spec: _VerifierVariantSpec) -> VerifierConfig:
    agent_config = config.get("agent") or {}
    if not isinstance(agent_config, dict):
        raise ValueError("Invalid config: 'agent' must be a mapping.")

    verifier_payload = copy.deepcopy(agent_config.get("verifier") or {})
    if not isinstance(verifier_payload, dict):
        raise ValueError("Invalid config: 'agent.verifier' must be a mapping.")

    verifier_payload["enabled"] = True
    verifier_payload["verifier_type"] = variant_spec.verifier_type
    if variant_spec.prompt_name is not None:
        verifier_payload["prompt_name"] = variant_spec.prompt_name
    else:
        verifier_payload["prompt_name"] = None
    if variant_spec.prompt_dir is not None:
        verifier_payload["prompt_dir"] = variant_spec.prompt_dir
    for key, value in variant_spec.config_overrides.items():
        verifier_payload[key] = copy.deepcopy(value)
    verifier_config = VerifierConfig(**verifier_payload)
    if variant_spec.verifier_type != "first_valid":
        verifier_config.prompt_name = _align_prompt_name(verifier_config.prompt_name, variant_spec.verifier_type)
        verifier_config = resolve_verifier_runtime_config(verifier_config)
    return verifier_config


def _build_verifier_session(config: dict[str, Any], variant_spec: _VerifierVariantSpec) -> _VerifierSession:
    verifier_config = _resolve_variant_verifier_config(config, variant_spec)

    if variant_spec.verifier_type == "first_valid":
        verifier = FirstValidVerifier(verifier_config)
    elif verifier_config.model:
        verifier_config.model = _normalize_verifier_model_config(verifier_config.model)
        verifier_model = get_model(verifier_config.model.get("model_name"), verifier_config.model)
        verifier = LLMVerifier(verifier_model, verifier_config) if variant_spec.verifier_type == "llm" else RewardModelVerifier(
            verifier_model,
            verifier_config,
        )
    else:
        actor_model_config = config.get("model") or {}
        if not isinstance(actor_model_config, dict):
            raise ValueError("Invalid config: 'model' must be a mapping.")
        verifier_model = get_model(config=actor_model_config)
        _assert_safe_verifier_fallback_model(verifier_model)
        verifier = LLMVerifier(verifier_model, verifier_config) if variant_spec.verifier_type == "llm" else RewardModelVerifier(
            verifier_model,
            verifier_config,
        )

    return _VerifierSession(
        variant_name=variant_spec.name,
        verifier_type=variant_spec.verifier_type,
        config=verifier_config,
        verifier=verifier,
    )


def _is_assistant_message(message: dict[str, Any]) -> bool:
    if message.get("role") == "assistant":
        return True
    if message.get("object") == "response":
        return True
    if message.get("type") == "message" and message.get("role") == "assistant":
        return True
    return False


def _redact_textual_content(content: Any) -> Any:
    if isinstance(content, str):
        return ""
    if isinstance(content, list):
        redacted_items: list[Any] = []
        for item in content:
            if isinstance(item, str):
                redacted_items.append("")
                continue
            if isinstance(item, dict):
                item_copy = copy.deepcopy(item)
                if isinstance(item_copy.get("text"), str):
                    item_copy["text"] = ""
                if "content" in item_copy:
                    item_copy["content"] = _redact_textual_content(item_copy.get("content"))
                redacted_items.append(item_copy)
                continue
            redacted_items.append(item)
        return redacted_items
    return content


def _redact_assistant_message_content(message: dict[str, Any]) -> dict[str, Any]:
    message_copy = copy.deepcopy(message)
    if "content" in message_copy:
        message_copy["content"] = _redact_textual_content(message_copy.get("content"))
    if isinstance(message_copy.get("output_text"), str):
        message_copy["output_text"] = ""

    output = message_copy.get("output")
    if isinstance(output, list):
        sanitized_output: list[Any] = []
        for item in output:
            if not isinstance(item, dict):
                sanitized_output.append(item)
                continue
            item_copy = copy.deepcopy(item)
            if item_copy.get("type") == "message" and item_copy.get("role") == "assistant":
                if "content" in item_copy:
                    item_copy["content"] = _redact_textual_content(item_copy.get("content"))
            sanitized_output.append(item_copy)
        message_copy["output"] = sanitized_output
    return message_copy


def _messages_for_outbound_context(
    messages: list[dict[str, Any]], *, include_assistant_content: bool
) -> list[dict[str, Any]]:
    if include_assistant_content:
        return copy.deepcopy(messages)

    sanitized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if _is_assistant_message(message):
            sanitized_messages.append(_redact_assistant_message_content(message))
        else:
            sanitized_messages.append(copy.deepcopy(message))
    return sanitized_messages


def _is_system_message(message: dict[str, Any]) -> bool:
    if message.get("role") == "system":
        return True
    if message.get("type") == "message" and message.get("role") == "system":
        return True
    return False


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    steps: list[list[dict[str, Any]]] = []
    current_step: list[dict[str, Any]] = []
    started = False

    for message in messages:
        if _is_assistant_message(message):
            if current_step:
                steps.append(current_step)
            current_step = [message]
            started = True
            continue
        if started:
            current_step.append(message)

    if current_step:
        steps.append(current_step)
    return steps


def _slice_steps(steps: list[list[dict[str, Any]]], history_steps: int) -> list[list[dict[str, Any]]]:
    if history_steps < 0:
        return steps
    return steps[-history_steps:]


def _extract_task_from_user_message(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""

    pr_description_match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", text, re.DOTALL | re.IGNORECASE)
    if pr_description_match is not None:
        extracted = pr_description_match.group(1).strip()
        extracted = re.sub(r"^\s*Consider the following PR description:\s*", "", extracted, count=1, flags=re.IGNORECASE)
        if extracted:
            return extracted
    return text


def _extract_task(history_trajectory: list[dict[str, Any]], row: dict[str, Any]) -> str:
    for message in history_trajectory:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = get_content_string(message)
        if isinstance(text, str) and text.strip():
            return _extract_task_from_user_message(text)

    for fallback_key in ("problem_id", "instance_id"):
        value = row.get(fallback_key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _build_candidate(action_entry: dict[str, Any], index: int) -> dict[str, Any]:
    command = action_entry.get("command")
    if not isinstance(command, str):
        command = ""

    action_payload: dict[str, Any] | None = None
    if command:
        action_payload = {"command": command}
        tool_call_id = action_entry.get("tool_call_id")
        if isinstance(tool_call_id, str):
            action_payload["tool_call_id"] = tool_call_id

    response_text = ""
    model_response = action_entry.get("model_response")
    if isinstance(model_response, dict):
        response_text = get_content_string(model_response).strip()

    candidate = {
        "index": index,
        "content": response_text or command or "(empty candidate)",
        "action": command or None,
        "actions": [action_payload] if action_payload else [],
        "n_actions": 1 if action_payload else 0,
        "label": action_entry.get("label"),
    }
    return candidate


def _is_gold_action(action_entry: dict[str, Any]) -> bool:
    if action_entry.get("is_gold") is True:
        return True
    if action_entry.get("candidate_source") == "gold":
        return True
    if action_entry.get("sampler_model_id") == "gold":
        return True
    return action_entry.get("label") == "gold"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sum_verifier_cost(verifier_output: dict[str, Any]) -> float:
    total = 0.0
    total += _safe_float(verifier_output.get("cost"))
    total += _safe_float(verifier_output.get("response_cost"))

    response_costs = verifier_output.get("response_costs")
    if isinstance(response_costs, list):
        for cost in response_costs:
            total += _safe_float(cost)

    checklist = verifier_output.get("checklist")
    if isinstance(checklist, dict):
        total += _safe_float(checklist.get("response_cost"))
    return total


def _sum_verifier_api_calls(verifier_output: dict[str, Any]) -> int:
    total = _safe_int(verifier_output.get("api_calls"))
    checklist = verifier_output.get("checklist")
    if isinstance(checklist, dict):
        total += _safe_int(checklist.get("api_calls"))
    return total


def _should_use_checklist_mode(config: VerifierConfig) -> bool:
    return verifier_uses_checklist_mode(config)


def _get_static_seed_checklist_config(checklist_config: VerifierConfig) -> VerifierConfig:
    variant_root = "checklist_v2" if resolve_checklist_output_format(checklist_config) == "rubric_yaml" else "checklist"
    static_prompt_name: str | None = None
    if checklist_config.verifier_type == "llm":
        static_prompt_name = f"{variant_root}/verifier"
    elif checklist_config.verifier_type == "reward_model":
        static_prompt_name = f"{variant_root}/reward"
    if static_prompt_name is None:
        return checklist_config
    static_config = checklist_config.model_copy(deep=True)
    static_config.prompt_name = static_prompt_name
    return resolve_verifier_runtime_config(static_config)


def _prepare_checklist_template_vars(
    session: _VerifierSession,
    run_key: tuple[str, str, str],
    template_vars: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checklist_config = session.config
    if not _should_use_checklist_mode(checklist_config):
        return template_vars, None

    checklist_data = session.checklist_cache.get(run_key)
    dynamic_enabled = bool(getattr(checklist_config, "checklist_dynamic", False))
    update_mode = getattr(checklist_config, "checklist_update_mode", "regenerate")

    previous_items = (
        [item.strip() for item in checklist_data.get("items", []) if isinstance(item, str) and item.strip()]
        if isinstance(checklist_data, dict)
        else []
    )
    previous_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(previous_items))
    generated_this_row = False
    seeded_from_static_prompt = False

    should_generate = dynamic_enabled or checklist_data is None or not bool(checklist_config.checklist_generate_once)
    if should_generate:
        generation_config = checklist_config
        if dynamic_enabled and update_mode == "modify" and not previous_items:
            generation_config = _get_static_seed_checklist_config(checklist_config)
            seeded_from_static_prompt = True

        generation_template_vars = {
            **template_vars,
            "checklist_update_mode": update_mode,
            "previous_checklist_items": previous_items,
            "previous_checklist_text": previous_text,
        }
        if resolve_checklist_output_format(generation_config) != "rubric_yaml":
            generation_template_vars["checklist_min_items"] = generation_config.checklist_min_items
            generation_template_vars["checklist_max_items"] = generation_config.checklist_max_items

        model = getattr(session.verifier, "model", None)
        if model is None:
            raise RuntimeError("Checklist mode requires verifier.model to be available.")

        checklist_data = generate_issue_checklist(
            model,
            generation_config,
            template_vars=generation_template_vars,
        )
        session.checklist_cache[run_key] = checklist_data
        generated_this_row = True

    raw_items = checklist_data.get("items", []) if isinstance(checklist_data, dict) else []
    checklist_items = [item.strip() for item in raw_items if isinstance(item, str) and item.strip()]
    raw_rubric_items = checklist_data.get("rubric_items", []) if isinstance(checklist_data, dict) else []
    checklist_rubric = [item for item in raw_rubric_items if isinstance(item, dict)]
    checklist_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(checklist_items))
    checklist_output_format = (
        checklist_data.get("checklist_output_format")
        if isinstance(checklist_data, dict)
        else resolve_checklist_output_format(checklist_config)
    )

    updated_template_vars = dict(template_vars)
    updated_template_vars.update(
        {
            "checklist_items": checklist_items,
            "checklist_text": checklist_text,
            "checklist_count": len(checklist_items),
            "checklist_rubric": checklist_rubric,
        }
    )

    checklist_metadata = {
        "items": checklist_items,
        "rubric_items": checklist_rubric,
        "checklist_output_format": checklist_output_format,
        "raw_output": checklist_data.get("raw_output", "") if isinstance(checklist_data, dict) else "",
        "input": checklist_data.get("input") if isinstance(checklist_data, dict) else None,
        "response": checklist_data.get("response", {}) if isinstance(checklist_data, dict) else {},
        "response_cost": _safe_float(checklist_data.get("response_cost")) if isinstance(checklist_data, dict) else 0.0,
        "api_calls": _safe_int(checklist_data.get("api_calls")) if isinstance(checklist_data, dict) else 0,
        "generated_once": checklist_config.checklist_generate_once,
        "generated_this_row": generated_this_row,
        "dynamic": dynamic_enabled,
        "update_mode": update_mode if dynamic_enabled else None,
        "generation_mode": "dynamic" if dynamic_enabled else "static",
        "source": (
            "issue_description"
            if not dynamic_enabled
            else "static_checklist_seed"
            if seeded_from_static_prompt
            else "dynamic_checklist"
        ),
    }
    return updated_template_vars, checklist_metadata


def _evaluate_row(
    *,
    row: dict[str, Any],
    row_index: int,
    line_no: int,
    session: _VerifierSession,
    strict_five_actions: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "row_index": row_index,
        "line_no": line_no,
        "verifier_variant": session.variant_name,
        "verifier_type": session.verifier_type,
        "instance_id": row.get("instance_id"),
        "run_id": row.get("run_id"),
        "trajectory_relpath": row.get("trajectory_relpath"),
        "step_index": row.get("step_index"),
        "message_index": row.get("message_index"),
    }

    actions_raw = row.get("actions")
    if not isinstance(actions_raw, list):
        output.update({"status": "skipped", "skip_reason": "missing_actions_list"})
        return output

    output["n_actions"] = len(actions_raw)
    if strict_five_actions and len(actions_raw) != 5:
        output.update({"status": "skipped", "skip_reason": "not_5_actions"})
        return output

    candidates: list[dict[str, Any]] = []
    candidate_labels: list[str | None] = []
    gold_indices: list[int] = []

    for idx, action_entry in enumerate(actions_raw):
        if not isinstance(action_entry, dict):
            output.update({"status": "skipped", "skip_reason": "malformed_action_entry"})
            return output
        candidate = _build_candidate(action_entry, len(candidates))
        candidates.append(candidate)
        label = action_entry.get("label") if isinstance(action_entry.get("label"), str) else None
        candidate_labels.append(label)
        if _is_gold_action(action_entry):
            gold_indices.append(len(candidates) - 1)

    if not candidates:
        output.update({"status": "skipped", "skip_reason": "no_candidates"})
        return output
    if len(gold_indices) != 1:
        output.update({"status": "skipped", "skip_reason": "gold_label_count_not_1", "gold_count": len(gold_indices)})
        return output

    gold_index = gold_indices[0]
    output["gold_index"] = gold_index
    output["candidate_labels"] = candidate_labels

    history_trajectory = row.get("history_trajectory")
    if not isinstance(history_trajectory, list):
        history_trajectory = []

    include_thoughts = bool(session.config.include_thoughts_in_history_steps)
    outbound_messages = _messages_for_outbound_context(
        [message for message in history_trajectory if isinstance(message, dict)],
        include_assistant_content=include_thoughts,
    )
    outbound_messages = [message for message in outbound_messages if not _is_system_message(message)]
    all_steps = _messages_to_steps(outbound_messages)
    history_steps = int(session.config.history_steps)
    steps = _slice_steps(all_steps, history_steps)

    messages = [message for step in steps for message in step]
    all_messages = [message for step in all_steps for message in step]
    task = _extract_task(outbound_messages, row)

    template_vars: dict[str, Any] = {
        "task": task,
        "messages": messages,
        "all_messages": all_messages,
        "steps": steps,
        "all_steps": all_steps,
        "history_steps": history_steps,
    }

    run_key = (
        str(row.get("instance_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("trajectory_relpath") or ""),
    )
    template_vars, checklist_metadata = _prepare_checklist_template_vars(session, run_key, template_vars)

    try:
        if session.verifier_type == "reward_model":
            selected_index, verifier_output = session.verifier.select(
                candidates=candidates,
                template_vars=template_vars,
                task=task,
                messages=messages,
                steps=steps,
            )
        else:
            selected_index, verifier_output = session.verifier.select(
                candidates=candidates,
                template_vars=template_vars,
            )
    except Exception as exc:  # noqa: BLE001
        output.update(
            {
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        return output

    if not isinstance(verifier_output, dict):
        verifier_output = {"raw_output": verifier_output}
    if checklist_metadata is not None:
        verifier_output = dict(verifier_output)
        verifier_output["checklist"] = checklist_metadata

    selected_index = max(0, min(int(selected_index), len(candidates) - 1))
    selected_label = candidate_labels[selected_index] if selected_index < len(candidate_labels) else None
    selected_is_gold = selected_index == gold_index

    output.update(
        {
            "status": "evaluated",
            "selected_index": selected_index,
            "selected_label": selected_label,
            "selected_is_gold": selected_is_gold,
            "history_steps": history_steps,
            "history_messages": len(messages),
            "include_thoughts_in_history_steps": include_thoughts,
            "verifier_output": verifier_output,
            "row_cost": _sum_verifier_cost(verifier_output),
            "row_api_calls": _sum_verifier_api_calls(verifier_output),
        }
    )
    return output


def _init_metric_bucket() -> dict[str, Any]:
    return {
        "rows_total": 0,
        "rows_evaluated": 0,
        "rows_skipped": 0,
        "rows_failed": 0,
        "gold_pick_count": 0,
        "accuracy": 0.0,
        "total_cost": 0.0,
        "total_api_calls": 0,
        "skip_reasons": {},
        "failure_reasons": {},
    }


def _build_failed_task_result(
    *,
    row: dict[str, Any],
    row_index: int,
    line_no: int,
    verifier_variant: str,
    verifier_type: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "line_no": line_no,
        "verifier_variant": verifier_variant,
        "verifier_type": verifier_type,
        "instance_id": row.get("instance_id"),
        "run_id": row.get("run_id"),
        "trajectory_relpath": row.get("trajectory_relpath"),
        "step_index": row.get("step_index"),
        "message_index": row.get("message_index"),
        "status": "failed",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _record_metric_row(
    *,
    metric: dict[str, Any],
    skip_counter: Counter,
    failure_counter: Counter,
    row_result: dict[str, Any],
) -> None:
    metric["rows_total"] += 1
    status = row_result.get("status")
    if status == "evaluated":
        metric["rows_evaluated"] += 1
        if row_result.get("selected_is_gold") is True:
            metric["gold_pick_count"] += 1
        metric["total_cost"] += _safe_float(row_result.get("row_cost"))
        metric["total_api_calls"] += _safe_int(row_result.get("row_api_calls"))
        return

    if status == "skipped":
        metric["rows_skipped"] += 1
        skip_counter[str(row_result.get("skip_reason") or "unknown")] += 1
        return

    metric["rows_failed"] += 1
    error_type = "unknown"
    error = row_result.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        error_type = error["type"]
    failure_counter[error_type] += 1


def _finalize_metric_bucket(metric: dict[str, Any], skip_counter: Counter, failure_counter: Counter) -> None:
    evaluated = _safe_int(metric.get("rows_evaluated"))
    gold_pick_count = _safe_int(metric.get("gold_pick_count"))
    metric["accuracy"] = (gold_pick_count / evaluated) if evaluated > 0 else 0.0
    metric["skip_reasons"] = dict(skip_counter)
    metric["failure_reasons"] = dict(failure_counter)


def evaluate_verifier_action_selection(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    output_summary: Path | None = None,
    config_specs: list[str] | None = None,
    verifier_types: list[str] | None = None,
    verifier_variants: list[str] | None = None,
    strict_five_actions: bool = True,
    limit_rows: int | None = None,
    show_progress: bool = True,
    max_workers: int = 8,
    enable_langfuse: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_summary is None:
        output_summary = output_jsonl.with_name("evaluation_summary.json")
    if not overwrite and output_jsonl.exists():
        raise FileExistsError(f"Output file already exists: {output_jsonl}")
    if not overwrite and output_summary.exists():
        raise FileExistsError(f"Summary file already exists: {output_summary}")

    resolved_variants = _resolve_verifier_variants(verifier_types, verifier_variants)
    resolved_verifier_types = list(dict.fromkeys(spec.verifier_type for spec in resolved_variants))
    resolved_config, resolved_specs = _load_resolved_config(config_specs)
    langfuse_session_id: str | None = None
    if enable_langfuse:
        enable_langfuse_tracing()
        langfuse_session_id = make_langfuse_session_id(
            prefix="verifier_eval",
            output_path=output_jsonl,
            parts=[input_jsonl.stem],
        )
        attach_langfuse_session_metadata(resolved_config, session_id=langfuse_session_id)
    requested_max_workers = max(1, int(max_workers))

    counts = {
        "input_rows": 0,
        "parsed_rows": 0,
        "invalid_rows": 0,
        "rows_considered": 0,
        "rows_written": 0,
    }
    per_variant_metrics = {spec.name: _init_metric_bucket() for spec in resolved_variants}
    per_variant_skip_counters = {spec.name: Counter() for spec in resolved_variants}
    per_variant_failure_counters = {spec.name: Counter() for spec in resolved_variants}
    per_type_metrics = {verifier_type: _init_metric_bucket() for verifier_type in resolved_verifier_types}
    per_type_skip_counters = {verifier_type: Counter() for verifier_type in resolved_verifier_types}
    per_type_failure_counters = {verifier_type: Counter() for verifier_type in resolved_verifier_types}

    rows_to_evaluate: list[tuple[int, int, dict[str, Any]]] = []
    with input_jsonl.open("r", encoding="utf-8") as input_handle:
        for line_no, raw_line in enumerate(input_handle, start=1):
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
            if limit_rows is not None and len(rows_to_evaluate) >= limit_rows:
                break
            rows_to_evaluate.append((line_no, len(rows_to_evaluate), row))

    counts["rows_considered"] = len(rows_to_evaluate)
    variant_order = {spec.name: idx for idx, spec in enumerate(resolved_variants)}
    tasks = [
        (line_no, row_index, row, spec)
        for line_no, row_index, row in rows_to_evaluate
        for spec in resolved_variants
    ]
    task_count = len(tasks)
    effective_max_workers = min(requested_max_workers, task_count) if task_count > 0 else 1
    if _requires_serial_evaluation(resolved_config, resolved_variants):
        effective_max_workers = 1

    progress = None
    if show_progress and _tqdm is not None:
        progress = _tqdm(
            total=task_count,
            desc="Evaluating verifier tasks",
            unit="task",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    shared_sessions: dict[str, _VerifierSession] | None = None
    thread_local = threading.local()

    def _evaluate_task(task: tuple[int, int, dict[str, Any], _VerifierVariantSpec]) -> dict[str, Any]:
        line_no, row_index, row, variant_spec = task
        try:
            if effective_max_workers <= 1:
                if shared_sessions is None:  # pragma: no cover - guarded by outer initialization
                    raise RuntimeError("shared verifier sessions are not initialized")
                session = shared_sessions[variant_spec.name]
            else:
                thread_sessions = getattr(thread_local, "sessions", None)
                if thread_sessions is None:
                    thread_sessions = {spec.name: _build_verifier_session(resolved_config, spec) for spec in resolved_variants}
                    thread_local.sessions = thread_sessions
                session = thread_sessions[variant_spec.name]

            return _evaluate_row(
                row=row,
                row_index=row_index,
                line_no=line_no,
                session=session,
                strict_five_actions=strict_five_actions,
            )
        except Exception as exc:  # noqa: BLE001
            return _build_failed_task_result(
                row=row,
                row_index=row_index,
                line_no=line_no,
                verifier_variant=variant_spec.name,
                verifier_type=variant_spec.verifier_type,
                exc=exc,
            )

    results: list[dict[str, Any]] = []
    try:
        if effective_max_workers <= 1:
            shared_sessions = {spec.name: _build_verifier_session(resolved_config, spec) for spec in resolved_variants}
            for task in tasks:
                results.append(_evaluate_task(task))
                if progress is not None:
                    progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
                futures = [executor.submit(_evaluate_task, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    results.sort(
        key=lambda row_result: (
            _safe_int(row_result.get("row_index")),
            variant_order.get(str(row_result.get("verifier_variant") or ""), 999),
            _safe_int(row_result.get("line_no")),
        )
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output_handle:
        for row_result in results:
            output_handle.write(json.dumps(row_result, ensure_ascii=False, default=str))
            output_handle.write("\n")
            counts["rows_written"] += 1

            verifier_variant = str(row_result.get("verifier_variant") or "")
            verifier_type = str(row_result.get("verifier_type") or "")
            if verifier_variant not in per_variant_metrics or verifier_type not in per_type_metrics:
                continue
            _record_metric_row(
                metric=per_variant_metrics[verifier_variant],
                skip_counter=per_variant_skip_counters[verifier_variant],
                failure_counter=per_variant_failure_counters[verifier_variant],
                row_result=row_result,
            )
            _record_metric_row(
                metric=per_type_metrics[verifier_type],
                skip_counter=per_type_skip_counters[verifier_type],
                failure_counter=per_type_failure_counters[verifier_type],
                row_result=row_result,
            )

    for variant_name, metric in per_variant_metrics.items():
        _finalize_metric_bucket(metric, per_variant_skip_counters[variant_name], per_variant_failure_counters[variant_name])
    for verifier_type, metric in per_type_metrics.items():
        _finalize_metric_bucket(metric, per_type_skip_counters[verifier_type], per_type_failure_counters[verifier_type])

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_summary": str(output_summary),
        "config_specs": resolved_specs,
        "verifier_variants": [spec.name for spec in resolved_variants],
        "verifier_types": resolved_verifier_types,
        "strict_five_actions": strict_five_actions,
        "limit_rows": limit_rows,
        "show_progress": show_progress,
        "max_workers": requested_max_workers,
        "effective_max_workers": effective_max_workers,
        "langfuse_session_id": langfuse_session_id,
        "counts": counts,
        "per_variant": per_variant_metrics,
        "per_verifier": per_type_metrics,
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary

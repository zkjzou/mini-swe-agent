from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.verifier_action_sampling import (
    extract_actions_from_assistant_message,
    normalize_docent_message_for_model,
)


class ReplayError(RuntimeError):
    """Raised when a logged prefix cannot be replayed safely."""


@dataclass(frozen=True)
class ReplaySummary:
    task: str
    replayed_prefix_steps: int
    seeded_message_count: int


def extract_task_from_history(history_trajectory: list[dict[str, Any]], row: dict[str, Any]) -> str:
    for message in history_trajectory:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = get_content_string(message)
        if content:
            return content
    for key in ("problem_statement", "problem_id", "instance_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _normalize_tool_call(command: str, *, tool_call_id: str | None) -> dict[str, Any]:
    return {
        "id": tool_call_id or f"call_mc_{int(time.time() * 1_000_000)}",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": command}, ensure_ascii=False),
        },
    }


def normalize_replay_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_docent_message_for_model(message)
    role = normalized.get("role")
    if role == "assistant":
        actions = extract_actions_from_assistant_message(message)
        if actions:
            normalized["extra"] = {
                **dict(normalized.get("extra", {}) or {}),
                "actions": actions,
            }
    return normalized


def candidate_thought(action: dict[str, Any]) -> str:
    direct = action.get("thought")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    model_response = action.get("model_response")
    if isinstance(model_response, dict):
        return flatten_message_content(model_response.get("content"))
    return ""


def candidate_label(action: dict[str, Any], action_index: int) -> str:
    raw = action.get("label")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"candidate_{action_index}"


def safe_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return sanitized or "candidate"


def build_candidate_branch_message(action: dict[str, Any], *, action_index: int) -> dict[str, Any]:
    command = str(action.get("command") or "").strip()
    if not command:
        raise ReplayError(f"Candidate {action_index} is missing a command.")

    tool_call_id = action.get("tool_call_id")
    tool_call_id_str = str(tool_call_id) if tool_call_id is not None else None
    model_response = action.get("model_response")
    if isinstance(model_response, dict):
        message = normalize_docent_message_for_model(model_response)
    else:
        message = {"role": "assistant", "content": candidate_thought(action)}

    message["role"] = "assistant"
    message["tool_calls"] = [_normalize_tool_call(command, tool_call_id=tool_call_id_str)]
    message["extra"] = {
        **dict(message.get("extra", {}) or {}),
        "actions": [{"command": command, **({"tool_call_id": tool_call_id_str} if tool_call_id_str else {})}],
        "candidate": {
            "label": candidate_label(action, action_index),
            "is_gold": bool(action.get("is_gold")),
            "sample_index": action.get("sample_index"),
            "candidate_source": action.get("candidate_source"),
        },
    }
    return message


def seed_agent_from_history(agent: Any, row: dict[str, Any]) -> ReplaySummary:
    history = row.get("history_trajectory")
    if not isinstance(history, list):
        raise ReplayError("Row is missing a valid history_trajectory list.")

    task = extract_task_from_history([msg for msg in history if isinstance(msg, dict)], row)
    agent.messages = []
    agent.step_count = 0
    agent.cost = 0.0
    agent.base_model_cost = 0.0
    agent.verifier_cost = 0.0
    agent.n_calls = 0
    agent.agent_api_calls = 0
    agent.verifier_api_calls = 0
    agent.checklist_api_calls = 0
    agent._verifier_checklist_cache = None
    agent._previous_verifier_feedback = None
    agent.extra_template_vars = {"task": task}

    replayed_prefix_steps = 0
    for original_message in history:
        if not isinstance(original_message, dict):
            continue
        role = original_message.get("role")
        if role == "tool":
            continue

        normalized = normalize_replay_message(original_message)
        if role != "assistant":
            agent.add_messages(normalized)
            continue

        actions = list((normalized.get("extra", {}) or {}).get("actions", []) or [])
        if len(actions) > 1:
            raise ReplayError("Parallel tool calls are not supported for Monte Carlo replay.")

        agent.add_messages(normalized)
        if not actions:
            continue

        try:
            agent.execute_actions(normalized)
        except InterruptAgentFlow as exc:
            agent.add_messages(*exc.messages)
            raise ReplayError(f"Replay hit terminal control flow unexpectedly: {type(exc).__name__}") from exc
        replayed_prefix_steps += 1

    return ReplaySummary(
        task=task,
        replayed_prefix_steps=replayed_prefix_steps,
        seeded_message_count=len(agent.messages),
    )

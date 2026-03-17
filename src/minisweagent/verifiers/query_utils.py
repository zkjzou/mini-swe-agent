from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS

_SINGLE_PROMPT_HISTORY_FORMAT = "single_prompt"
_MULTI_TURN_CHAT_HISTORY_FORMAT = "multi_turn_chat"


def query_verifier_text(model: Any, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], float]:
    """Query a verifier model for plain-text output without requiring action parsing."""
    if hasattr(model, "query_raw"):
        raw_response = model.query_raw(messages)
        response_cost = _calculate_cost(model, raw_response)
        GLOBAL_MODEL_STATS.add(response_cost)
        return _extract_text(raw_response), _serialize_response(raw_response), response_cost

    if not hasattr(model, "_query"):
        response = model.query(messages)
        return _extract_text(response), _serialize_response(response), _extract_cost_from_message(response)

    prepared_messages = messages
    if hasattr(model, "_prepare_messages_for_api"):
        prepared_messages = model._prepare_messages_for_api(messages)

    raw_response = model._query(prepared_messages)
    response_cost = _calculate_cost(model, raw_response)
    GLOBAL_MODEL_STATS.add(response_cost)
    return _extract_text(raw_response), _serialize_response(raw_response), response_cost


def resolve_verifier_history_message_format(config: Any) -> str:
    configured = getattr(config, "history_message_format", _SINGLE_PROMPT_HISTORY_FORMAT)
    if isinstance(configured, str) and configured.strip().lower() == _MULTI_TURN_CHAT_HISTORY_FORMAT:
        return _MULTI_TURN_CHAT_HISTORY_FORMAT
    return _SINGLE_PROMPT_HISTORY_FORMAT


def build_verifier_messages(
    *,
    config: Any,
    system_prompt: str,
    final_user_prompt: str,
    template_vars: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    template_vars = template_vars or {}
    history_format = resolve_verifier_history_message_format(config)
    input_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if history_format == _MULTI_TURN_CHAT_HISTORY_FORMAT:
        input_messages.extend(_extract_history_messages(template_vars))
    input_messages.append({"role": "user", "content": final_user_prompt})
    return sanitize_captured_verifier_messages(
        input_messages,
        allow_assistant=history_format == _MULTI_TURN_CHAT_HISTORY_FORMAT,
        allow_tool=history_format == _MULTI_TURN_CHAT_HISTORY_FORMAT,
    )["messages"]


def sanitize_captured_verifier_messages(
    messages: list[dict[str, Any]], *, allow_assistant: bool = False, allow_tool: bool = False
) -> dict[str, Any]:
    allowed_roles = {"system", "user"}
    if allow_assistant:
        allowed_roles.add("assistant")
    if allow_tool:
        allowed_roles.add("tool")
    sanitized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _extract_history_role(message)
        if role not in allowed_roles:
            continue
        sanitized = _sanitize_history_message(message, role=role)
        if sanitized is None:
            continue
        sanitized_messages.append(sanitized)
    return {"messages": sanitized_messages}


def _extract_history_messages(template_vars: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_messages = template_vars.get("messages")
    if not isinstance(raw_messages, list):
        raw_steps = template_vars.get("steps")
        if isinstance(raw_steps, list):
            raw_messages = [message for step in raw_steps if isinstance(step, list) for message in step]
        else:
            raw_messages = []
    history_messages: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = _extract_history_role(message)
        if role not in {"user", "assistant", "tool"}:
            continue
        sanitized = _sanitize_history_message(message, role=role)
        if sanitized is None:
            continue
        history_messages.append(sanitized)
    return history_messages


def _extract_history_role(message: Mapping[str, Any]) -> str | None:
    role = message.get("role")
    if role in {"system", "user", "assistant", "tool"}:
        return role
    if message.get("type") == "function_call_output":
        return "tool"
    if message.get("type") == "message":
        typed_role = message.get("role")
        if typed_role in {"system", "user", "assistant", "tool"}:
            return typed_role
    return None


def _sanitize_history_message(message: Mapping[str, Any], *, role: str) -> dict[str, Any] | None:
    if role == "assistant":
        return _sanitize_assistant_history_message(message)
    if role == "tool":
        return _sanitize_tool_history_message(message)
    content = _extract_message_text(message)
    if not content:
        return None
    return {"role": role, "content": content}


def _sanitize_assistant_history_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    content = _extract_message_text(message)
    tool_calls = _extract_tool_calls(message)
    if not content and not tool_calls:
        return None
    sanitized: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        sanitized["tool_calls"] = tool_calls
    return sanitized


def _sanitize_tool_history_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    content = _extract_message_text(message)
    if not content:
        return None
    sanitized: dict[str, Any] = {"role": "tool", "content": content}
    tool_call_id = message.get("tool_call_id")
    if not isinstance(tool_call_id, str):
        call_id = message.get("call_id")
        if isinstance(call_id, str):
            tool_call_id = call_id
    if isinstance(tool_call_id, str) and tool_call_id:
        sanitized["tool_call_id"] = tool_call_id
    name = message.get("name")
    if isinstance(name, str) and name:
        sanitized["name"] = name
    return sanitized


def _extract_message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _extract_text_from_content(content)
        if text:
            return text
    if isinstance(message.get("output_text"), str):
        return message["output_text"]
    output = message.get("output")
    if isinstance(output, list):
        return _extract_text_from_output(output)
    return ""


def _extract_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized_tool_calls: list[dict[str, Any]] = []
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for tool_call in raw_tool_calls:
            normalized = _normalize_tool_call(tool_call)
            if normalized is not None:
                normalized_tool_calls.append(normalized)
    output = message.get("output")
    if isinstance(output, list):
        for item in output:
            item_dict = _to_dict(item)
            if item_dict.get("type") == "function_call":
                normalized = _normalize_tool_call(item_dict)
                if normalized is not None:
                    normalized_tool_calls.append(normalized)
    if normalized_tool_calls:
        return normalized_tool_calls
    extra = message.get("extra")
    if isinstance(extra, Mapping):
        actions = extra.get("actions")
        if isinstance(actions, list):
            for action in actions:
                normalized = _normalize_action_tool_call(action)
                if normalized is not None:
                    normalized_tool_calls.append(normalized)
    return normalized_tool_calls


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    tool_call_dict = _to_dict(tool_call)
    if not tool_call_dict:
        return None
    function = tool_call_dict.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
        tool_call_id = tool_call_dict.get("id")
    else:
        name = tool_call_dict.get("name")
        arguments = tool_call_dict.get("arguments")
        tool_call_id = tool_call_dict.get("call_id") or tool_call_dict.get("id")
    if not isinstance(name, str) or not name:
        return None
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if arguments is not None else "",
        },
    }
    if isinstance(tool_call_id, str) and tool_call_id:
        normalized["id"] = tool_call_id
    return normalized


def _normalize_action_tool_call(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None
    command = action.get("command")
    if not isinstance(command, str) or not command:
        return None
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}},
    }
    tool_call_id = action.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        normalized["id"] = tool_call_id
    return normalized


def _calculate_cost(model: Any, response: Any) -> float:
    if not hasattr(model, "_calculate_cost"):
        return 0.0
    cost_output = model._calculate_cost(response)
    if isinstance(cost_output, Mapping):
        return float(cost_output.get("cost", 0.0) or 0.0)
    return float(cost_output or 0.0)


def _extract_cost_from_message(response: Any) -> float:
    if not isinstance(response, Mapping):
        return 0.0
    extra = response.get("extra", {})
    if not isinstance(extra, Mapping):
        return 0.0
    return float(extra.get("cost", 0.0) or 0.0)


def _serialize_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(response, "dict"):
        dumped = response.dict()
        if isinstance(dumped, dict):
            return dumped
    return {"repr": repr(response)}


def _extract_text(response: Any) -> str:
    text = _extract_text_from_dict(_serialize_response(response))
    if text:
        return text
    if isinstance(response, Mapping):
        return ""
    if hasattr(response, "output_text") and isinstance(response.output_text, str):
        return response.output_text
    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        text = _extract_text_from_choices(choices)
        if text:
            return text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        return _extract_text_from_output(output)
    return ""


def _extract_text_from_dict(response: dict[str, Any]) -> str:
    if isinstance(response.get("content"), str):
        return response["content"]
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    content = response.get("content")
    text = _extract_text_from_content(content)
    if text:
        return text

    choices = response.get("choices")
    if isinstance(choices, list):
        text = _extract_text_from_choices(choices)
        if text:
            return text

    output = response.get("output")
    if isinstance(output, list):
        text = _extract_text_from_output(output)
        if text:
            return text

    return ""


def _extract_text_from_choices(choices: list[Any]) -> str:
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, Mapping):
        message = first.get("message", {})
    elif hasattr(first, "message"):
        message = first.message
    else:
        return ""

    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return _extract_text_from_content(content)


def _extract_text_from_output(output: list[Any]) -> str:
    parts: list[str] = []
    for item in output:
        item_dict = _to_dict(item)
        if item_dict.get("type") == "message":
            text = _extract_text_from_content(item_dict.get("content"))
            if text:
                parts.append(text)
        elif item_dict.get("type") == "output_text":
            text = item_dict.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        item_dict = _to_dict(item)
        text = item_dict.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)




def _to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        dumped = item.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(item, "dict"):
        dumped = item.dict()
        if isinstance(dumped, dict):
            return dumped
    return {}

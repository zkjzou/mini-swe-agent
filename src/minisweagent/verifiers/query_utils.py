from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS


def query_verifier_text(model: Any, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], float]:
    """Query a verifier model for plain-text output without requiring action parsing."""
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

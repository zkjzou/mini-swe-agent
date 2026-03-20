from __future__ import annotations

from copy import deepcopy
from typing import Any

_REASONING_ITEM_TYPES = {"reasoning", "reasoning_text"}


def strip_reasoning_items(value: Any) -> Any:
    """Remove Responses API reasoning items from nested message payloads."""
    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in _REASONING_ITEM_TYPES:
                continue
            sanitized_item = strip_reasoning_items(item)
            if _should_drop_item(sanitized_item):
                continue
            sanitized_list.append(sanitized_item)
        return sanitized_list

    if isinstance(value, dict):
        sanitized = deepcopy(value)
        if "content" in sanitized:
            sanitized["content"] = strip_reasoning_items(sanitized.get("content"))
        if "output" in sanitized:
            sanitized["output"] = strip_reasoning_items(sanitized.get("output"))
        return sanitized

    return value


def _should_drop_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if item_type in _REASONING_ITEM_TYPES:
        return True
    if item_type == "message":
        content = item.get("content")
        tool_calls = item.get("tool_calls")
        return content == [] and not tool_calls
    return False

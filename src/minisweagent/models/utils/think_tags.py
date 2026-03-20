from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(value: Any) -> Any:
    """Remove <think>...</think> blocks from nested message payloads."""
    if isinstance(value, str):
        return _strip_think_tags_from_text(value)

    if isinstance(value, list):
        return [strip_think_tags(item) for item in value]

    if isinstance(value, dict):
        sanitized = deepcopy(value)
        for key in ("content", "output_text", "text"):
            if key in sanitized:
                sanitized[key] = strip_think_tags(sanitized.get(key))
        if "output" in sanitized:
            sanitized["output"] = strip_think_tags(sanitized.get("output"))
        return sanitized

    return value


def _strip_think_tags_from_text(text: str) -> str:
    stripped = _THINK_TAG_RE.sub("", text)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()

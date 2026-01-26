from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_trajectory(path: Path) -> dict[str, Any]:
    """Load a trajectory file and return a normalized dict with messages and info."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {
            "messages": data,
            "info": {},
            "trajectory_format": "list",
        }
    if isinstance(data, dict) and "messages" in data:
        return {
            "messages": data.get("messages", []),
            "info": data.get("info", {}),
            "trajectory_format": data.get("trajectory_format"),
        }
    raise ValueError(f"Unrecognized trajectory format in {path}")


def messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages into steps, where each step ends with a user message."""
    steps: list[list[dict[str, Any]]] = []
    current_step: list[dict[str, Any]] = []
    for message in messages:
        current_step.append(message)
        if message.get("role") == "user":
            steps.append(current_step)
            current_step = []
    if current_step:
        steps.append(current_step)
    return steps


def build_message_history(
    messages: list[dict[str, Any]], *, include_thoughts: bool
) -> list[dict[str, Any]]:
    """Return a filtered message history for rollouts.

    When include_thoughts is False, assistant messages are omitted.
    """
    if include_thoughts:
        return [message.copy() for message in messages]
    return [message.copy() for message in messages if message.get("role") != "assistant"]


def flatten_steps(steps: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten a list of steps back into a message list."""
    return [message for step in steps for message in step]

from __future__ import annotations

from typing import Any


class FirstValidVerifier:
    """Selects the first candidate that contains a valid action."""

    def __init__(self, config: Any):
        self.config = config

    def select(self, *, candidates: list[dict[str, Any]], template_vars: dict[str, Any] | None = None) -> tuple[int, dict]:
        selected_index = 0
        for candidate in candidates:
            if candidate.get("action"):
                selected_index = candidate["index"]
                break
        return selected_index, {"verifier_type": "first_valid"}

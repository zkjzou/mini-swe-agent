from __future__ import annotations

import re
from typing import Any

from jinja2 import StrictUndefined, Template


class LLMVerifier:
    """Uses an LLM to select the best candidate action."""

    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config

    def select(self, *, candidates: list[dict[str, Any]], template_vars: dict[str, Any] | None = None) -> tuple[int, dict]:
        template_vars = template_vars or {}
        system_prompt = self._render(
            self.config.system_template,
            candidates=candidates,
            selection_index_base=self.config.selection_index_base,
            **template_vars,
        )
        selection_prompt = self._render(
            self.config.selection_template,
            candidates=candidates,
            selection_index_base=self.config.selection_index_base,
            **template_vars,
        )
        response = self.model.query(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": selection_prompt},
            ]
        )
        content = response.get("content", "")
        matches = re.findall(self.config.selection_regex, content)
        selected_index = None
        raw_index = None
        if matches:
            raw_index = matches[-1]
            if isinstance(raw_index, tuple):
                raw_index = raw_index[0]
            raw_index = int(raw_index)
            selected_index = raw_index - self.config.selection_index_base
        if selected_index is None or not (0 <= selected_index < len(candidates)):
            selected_index = self._fallback_index(candidates)
        metadata = {
            "verifier_type": "llm",
            "raw_output": content,
            "raw_index": raw_index,
            "parsed_index": selected_index,
            "response": response,
        }
        return selected_index, metadata

    def _render(self, template: str, **kwargs) -> str:
        return Template(template, undefined=StrictUndefined).render(**kwargs)

    def _fallback_index(self, candidates: list[dict[str, Any]]) -> int:
        if self.config.fallback == "first_valid":
            for candidate in candidates:
                if candidate.get("action"):
                    return candidate["index"]
        return 0

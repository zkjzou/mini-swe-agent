from __future__ import annotations

import re
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent.verifiers.query_utils import query_verifier_text


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
        content, response, response_cost = query_verifier_text(
            self.model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": selection_prompt},
            ],
        )
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
        parsed_scores = self._parse_scores(content, len(candidates))
        n_checklist_items = self._count_checklist_items(template_vars)
        progress_score = self._parse_progress_score(content)
        checklist_item_scores = self._parse_checklist_item_scores(content, n_checklist_items)
        metadata = {
            "verifier_type": "llm",
            "raw_output": content,
            "raw_index": raw_index,
            "parsed_index": selected_index,
            "scores": parsed_scores,
            "progress_score": progress_score,
            "checklist_item_scores": checklist_item_scores,
            "response": response,
            "response_cost": response_cost,
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

    def _parse_scores(self, content: str, n_candidates: int) -> list[float | None]:
        score_regex = getattr(self.config, "selection_score_regex", r"Candidate\s+(\d+)\s*:\s*([+-]?\d+(?:\.\d+)?)")
        scores: list[float | None] = [None] * n_candidates
        for match in re.finditer(score_regex, content):
            try:
                raw_idx, raw_score = match.groups()
                idx = int(raw_idx) - self.config.selection_index_base
                score = float(raw_score)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(scores):
                scores[idx] = score
        return scores

    def _parse_progress_score(self, content: str) -> float | None:
        progress_regex = getattr(self.config, "checklist_progress_regex", r"PROGRESS:\s*([+-]?\d+(?:\.\d+)?)")
        matches = re.findall(progress_regex, content, re.MULTILINE)
        if not matches:
            return None
        raw_score = matches[-1]
        if isinstance(raw_score, tuple):
            raw_score = raw_score[0]
        try:
            return float(raw_score)
        except (ValueError, TypeError):
            return None

    def _parse_checklist_item_scores(self, content: str, n_items: int) -> list[float | None]:
        if n_items <= 0:
            return []
        item_score_regex = getattr(self.config, "checklist_item_score_regex", r"Item\s+(\d+)\s*:\s*([+-]?\d+(?:\.\d+)?)")
        scores: list[float | None] = [None] * n_items
        for match in re.finditer(item_score_regex, content, re.MULTILINE):
            groups = match.groups()
            if len(groups) < 2:
                continue
            try:
                raw_idx, raw_score = groups[0], groups[1]
                idx = int(raw_idx) - 1
                score = float(raw_score)
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(scores):
                scores[idx] = score
        return scores

    def _count_checklist_items(self, template_vars: dict[str, Any]) -> int:
        checklist_items = template_vars.get("checklist_items")
        if not isinstance(checklist_items, list):
            return 0
        return len(checklist_items)

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent.verifiers.query_utils import query_verifier_text


class RewardModelVerifier:
    """Scores each candidate action with a reward model and picks the highest reward."""

    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config

    def select(
        self,
        *,
        candidates: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
        task: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        steps: list[list[dict[str, Any]]] | None = None,
    ) -> tuple[int, dict]:
        template_vars = template_vars or {}
        verifier_vars = dict(template_vars)
        verifier_vars["task"] = task or ""
        if messages is not None:
            verifier_vars["messages"] = messages
        if steps is not None:
            verifier_vars["steps"] = steps
        n_checklist_items = self._count_checklist_items(verifier_vars)

        def _score_candidate(candidate: dict[str, Any]) -> tuple[float | None, float | None, list[float | None], str, dict[str, Any], float]:
            system_prompt = self._render(
                self.config.reward_system_template,
                candidates=candidates,
                candidate=candidate,
                **verifier_vars,
            )
            reward_prompt = self._render(
                self.config.reward_prompt_template,
                candidates=candidates,
                candidate=candidate,
                **verifier_vars,
            )
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    content, response, response_cost = query_verifier_text(
                        self.model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": reward_prompt},
                        ],
                    )
                    reward = self._parse_reward(content)
                    progress_score = self._parse_progress_score(content)
                    checklist_item_scores = self._parse_checklist_item_scores(content, n_checklist_items)
                    return reward, progress_score, checklist_item_scores, content, response, response_cost
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(4**attempt)
            raise last_exc or RuntimeError("reward model query failed")

        rewards: list[float | None] = [None] * len(candidates)
        raw_outputs: list[str] = [""] * len(candidates)
        responses: list[dict[str, Any]] = [{} for _ in candidates]
        response_costs: list[float] = [0.0] * len(candidates)
        progress_scores: list[float | None] = [None] * len(candidates)
        checklist_item_scores_by_candidate: list[list[float | None]] = [[] for _ in candidates]
        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
            futures = {
                executor.submit(_score_candidate, candidate): idx for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx = futures[future]
                reward, progress_score, checklist_item_scores, content, response, response_cost = future.result()
                rewards[idx] = reward
                progress_scores[idx] = progress_score
                checklist_item_scores_by_candidate[idx] = checklist_item_scores
                raw_outputs[idx] = content
                responses[idx] = response
                response_costs[idx] = response_cost
        selected_index = self._select_best(rewards, candidates)
        metadata = {
            "verifier_type": "reward_model",
            "rewards": rewards,
            "candidate_progress_scores": progress_scores,
            "candidate_checklist_item_scores": checklist_item_scores_by_candidate,
            "raw_outputs": raw_outputs,
            "responses": responses,
            "response_costs": response_costs,
        }
        return selected_index, metadata

    def _render(self, template: str, **kwargs) -> str:
        return Template(template, undefined=StrictUndefined).render(**kwargs)

    def _parse_reward(self, content: str) -> float | None:
        matches = re.findall(self.config.reward_regex, content, re.MULTILINE)
        if not matches:
            return None
        raw = matches[-1]
        if isinstance(raw, tuple):
            raw = raw[0]
        try:
            return float(raw)
        except ValueError:
            return None

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

    def _parse_rewards(self, content: str, n_candidates: int) -> list[float | None]:
        matches = re.findall(self.config.reward_regex, content, re.MULTILINE)
        rewards: list[float | None] = []
        for raw in matches[:n_candidates]:
            if isinstance(raw, tuple):
                raw = raw[0]
            try:
                rewards.append(float(raw))
            except ValueError:
                rewards.append(None)
        if len(rewards) < n_candidates:
            rewards.extend([None] * (n_candidates - len(rewards)))
        return rewards

    def _select_best(self, rewards: list[float | None], candidates: list[dict[str, Any]]) -> int:
        best_index = None
        best_value = None
        for idx, reward in enumerate(rewards):
            if reward is None:
                continue
            if best_value is None or reward > best_value:
                best_value = reward
                best_index = idx
        if best_index is not None:
            return best_index
        if self.config.fallback == "first_valid":
            for candidate in candidates:
                if candidate.get("action"):
                    return candidate["index"]
        return 0

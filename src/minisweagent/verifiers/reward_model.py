from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from jinja2 import StrictUndefined, Template


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

        def _score_candidate(candidate: dict[str, Any]) -> tuple[float | None, str, dict[str, Any]]:
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
                    response = self.model.query(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": reward_prompt},
                        ]
                    )
                    content = response.get("content", "")
                    return self._parse_reward(content), content, response
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(4**attempt)
            raise last_exc or RuntimeError("reward model query failed")

        rewards: list[float | None] = [None] * len(candidates)
        raw_outputs: list[str] = [""] * len(candidates)
        responses: list[dict[str, Any]] = [{} for _ in candidates]
        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
            futures = {
                executor.submit(_score_candidate, candidate): idx for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx = futures[future]
                reward, content, response = future.result()
                rewards[idx] = reward
                raw_outputs[idx] = content
                responses[idx] = response
        selected_index = self._select_best(rewards, candidates)
        metadata = {
            "verifier_type": "reward_model",
            "rewards": rewards,
            "raw_outputs": raw_outputs,
            "responses": responses,
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

"""Utilities for sampling and storing rejected actions for verifier training."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from minisweagent.models import get_model
from minisweagent.utils.log import logger

DEFAULT_ACTION_REGEX = r"```bash\s*\n(.*?)\n```"
DEFAULT_SELECTION_POLICY = "round_robin"


@dataclass(frozen=True)
class RejectedModelSpec:
    model_name: str
    model_class: str
    model_kwargs: dict[str, Any]
    model_id: str


def build_model_id(model_name: str, model_class: str) -> str:
    if model_class:
        return f"{model_class}:{model_name}"
    return model_name


def normalize_model_pool(
    model_pool: list[dict[str, Any]],
    *,
    expert_model_name: str | None,
    expert_model_class: str | None,
) -> list[RejectedModelSpec]:
    normalized: list[RejectedModelSpec] = []
    for spec in model_pool:
        if not isinstance(spec, dict):
            raise ValueError("Rejected model spec must be a mapping")
        model_name = str(spec.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("Rejected model spec missing model_name")
        model_class = str(spec.get("model_class", "")).strip()
        model_kwargs = spec.get("model_kwargs", {}) or {}
        model_id = spec.get("model_id") or build_model_id(model_name, model_class)

        if expert_model_name and model_name == expert_model_name:
            logger.warning("Skipping rejected model '%s' (matches expert model name)", model_name)
            continue
        if expert_model_class and model_class and model_class == expert_model_class:
            logger.warning("Skipping rejected model '%s' (matches expert model class)", model_id)
            continue

        normalized.append(
            RejectedModelSpec(
                model_name=model_name,
                model_class=model_class,
                model_kwargs=model_kwargs,
                model_id=str(model_id),
            )
        )
    return normalized


def extract_action_from_response(content: str, action_regex: str) -> str:
    actions = re.findall(action_regex, content, re.DOTALL)
    if len(actions) != 1:
        raise ValueError(f"Expected exactly one action, found {len(actions)}")
    return actions[0].strip()


def resolve_output_path(base_path: Path, run_id: str | None) -> Path:
    if not run_id:
        return base_path
    return base_path.with_name(f"{base_path.stem}.{run_id}{base_path.suffix}")


def write_rejected_actions(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        mode = "w"
    elif path.exists():
        mode = "a"
    else:
        mode = "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class RejectedActionSampler:
    def __init__(
        self,
        *,
        model_pool: list[dict[str, Any]],
        action_regex: str = DEFAULT_ACTION_REGEX,
        selection_policy: str = DEFAULT_SELECTION_POLICY,
        k_per_step: int = 1,
        max_attempts_per_step: int | None = None,
        timeout_s: float | None = None,
        expert_model_name: str | None = None,
        expert_model_class: str | None = None,
        output_path: Path | None = None,
        overwrite: bool = False,
        run_id: str | None = None,
        mode: str = "online",
    ) -> None:
        self.action_regex = action_regex or DEFAULT_ACTION_REGEX
        self.selection_policy = selection_policy or DEFAULT_SELECTION_POLICY
        self.k_per_step = max(0, int(k_per_step))
        self.max_attempts_per_step = max_attempts_per_step
        self.timeout_s = timeout_s
        self.mode = mode

        self.model_specs = normalize_model_pool(
            model_pool,
            expert_model_name=expert_model_name,
            expert_model_class=expert_model_class,
        )
        self._models: dict[str, Any] = {}
        self._selector_index = 0

        self._output_path = resolve_output_path(output_path, run_id) if output_path else None
        self._overwrite = overwrite
        self._has_written = False

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def _select_model_spec(self) -> RejectedModelSpec:
        if not self.model_specs:
            raise RuntimeError("No rejected models configured")
        if self.selection_policy == "random":
            return random.choice(self.model_specs)
        if self.selection_policy != "round_robin":
            raise ValueError(f"Unknown selection policy: {self.selection_policy}")
        spec = self.model_specs[self._selector_index % len(self.model_specs)]
        self._selector_index += 1
        return spec

    def _get_model(self, spec: RejectedModelSpec):
        if spec.model_id not in self._models:
            config: dict[str, Any] = {"model_name": spec.model_name}
            if spec.model_class:
                config["model_class"] = spec.model_class
            if spec.model_kwargs:
                config["model_kwargs"] = spec.model_kwargs
            self._models[spec.model_id] = get_model(config=config)
        return self._models[spec.model_id]

    def sample_records(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        step_index: int,
        expert_action: str,
    ) -> list[dict[str, Any]]:
        if not self.model_specs or self.k_per_step <= 0:
            return []
        records: list[dict[str, Any]] = []
        max_attempts = self.max_attempts_per_step
        if max_attempts is None:
            max_attempts = max(self.k_per_step, self.k_per_step * 2)

        attempts = 0
        while len(records) < self.k_per_step and attempts < max_attempts:
            attempts += 1
            spec = self._select_model_spec()
            model = self._get_model(spec)
            try:
                if self.timeout_s:
                    response = model.query(prompt_messages, timeout_s=self.timeout_s)
                else:
                    response = model.query(prompt_messages)
                rejected_action = extract_action_from_response(response["content"], self.action_regex)
            except Exception as exc:  # pragma: no cover - depends on external model behavior
                logger.warning("Rejected action sampling failed for %s: %s", spec.model_id, exc)
                continue
            records.append(
                {
                    "step_index": step_index,
                    "expert_action": expert_action,
                    "rejected_action": rejected_action,
                    "rejected_model_id": spec.model_id,
                    "mode": self.mode,
                    "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                }
            )

        if len(records) < self.k_per_step:
            logger.warning(
                "Only sampled %s/%s rejected actions for step %s",
                len(records),
                self.k_per_step,
                step_index,
            )
        return records

    def sample_and_write(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        step_index: int,
        expert_action: str,
    ) -> None:
        if not self._output_path:
            return
        records = self.sample_records(
            prompt_messages=prompt_messages,
            step_index=step_index,
            expert_action=expert_action,
        )
        if not records:
            return
        overwrite = self._overwrite and not self._has_written
        write_rejected_actions(self._output_path, records, overwrite=overwrite)
        self._has_written = True

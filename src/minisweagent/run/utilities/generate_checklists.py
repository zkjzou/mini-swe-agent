#!/usr/bin/env python3

"""Compatibility wrapper for trajectory checklist generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minisweagent.agents.default import VerifierConfig
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.utils.serialize import UNSET, recursive_merge
from minisweagent.verifiers.checklist import generate_issue_checklist
from minisweagent.verifiers.checklist_generator import normalize_checklist_generator_model_config


def _load_messages(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        messages = payload.get("messages", [])
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    if isinstance(payload, list):
        return [message for message in payload if isinstance(message, dict)]
    raise ValueError(f"Unsupported trajectory payload in {path}")


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    relevant_messages = messages[2:] if len(messages) >= 2 and messages[0].get("role") == "system" else messages
    steps: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in relevant_messages:
        if message.get("role") == "assistant" and current:
            steps.append(current)
            current = [message]
            continue
        current.append(message)
    if current:
        steps.append(current)
    return steps


def _extract_task(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _build_config(
    *,
    config_specs: list[str],
    model_name: str | None,
    model_class: str | None,
    generator_mode: str,
    generator_prompt_name: str | None,
) -> VerifierConfig:
    configs = [get_config_from_spec(spec) for spec in config_specs]
    configs.append(
        {
            "agent": {
                "verifier": {
                    "model": {
                        "model_name": model_name or UNSET,
                        "model_class": model_class or UNSET,
                    },
                    "checklist_mode": "issue_progress",
                    "checklist_generator_mode": generator_mode,
                    "checklist_generator_prompt_name": generator_prompt_name or UNSET,
                    "checklist_output_format": "rubric_yaml",
                    "include_inputs_in_output": True,
                }
            }
        }
    )
    merged = recursive_merge(*configs)
    return VerifierConfig(**merged.get("agent", {}).get("verifier", {}))


def main(
    *,
    input_paths: list[Path],
    output: Path,
    model_name: str | None,
    model_class: str | None,
    generator_mode: str,
    generator_prompt_name: str | None,
    current_step: int | None,
    config_specs: list[str],
) -> None:
    verifier_config = _build_config(
        config_specs=config_specs,
        model_name=model_name,
        model_class=model_class,
        generator_mode=generator_mode,
        generator_prompt_name=generator_prompt_name,
    )
    verifier_config.model = normalize_checklist_generator_model_config(dict(verifier_config.model))
    verifier_model = get_model(config=dict(verifier_config.model))
    payload: list[dict[str, Any]] = []
    for path in input_paths:
        messages = _load_messages(path)
        steps = _messages_to_steps(messages)
        active_steps = steps
        if generator_mode == "trajectory_dynamic":
            if current_step is None:
                raise ValueError("current_step is required for trajectory_dynamic mode")
            active_steps = steps[: max(0, current_step - 1)]
        checklist = generate_issue_checklist(
            verifier_model,
            verifier_config,
            template_vars={
                "task": _extract_task(messages),
                "messages": [message for step in active_steps for message in step],
                "steps": active_steps,
                "all_steps": steps,
            },
        )
        payload.append({"trajectory_path": str(path), "checklist": checklist})
    output.write_text(json.dumps(payload, indent=2))

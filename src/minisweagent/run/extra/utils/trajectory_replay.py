from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from jinja2 import StrictUndefined, Template

from minisweagent import Environment
from minisweagent.run.utils.trajectory import build_message_history, flatten_steps, messages_to_steps


def _coerce_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content)
    return str(content or "")


@dataclass(frozen=True)
class SelectedAction:
    action: str
    source: str = "assistant"
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionSelector(Protocol):
    def select_action(self, step_messages: list[dict[str, Any]], *, step_index: int) -> SelectedAction: ...


class RegexActionSelector:
    def __init__(self, action_regex: str):
        self.action_regex = action_regex

    def select_action(self, step_messages: list[dict[str, Any]], *, step_index: int) -> SelectedAction:
        actions: list[str] = []
        for message in step_messages:
            if message.get("role") != "assistant":
                continue
            content = _coerce_content(message.get("content"))
            actions.extend(re.findall(self.action_regex, content, re.DOTALL))
        if len(actions) != 1:
            raise ValueError(f"Expected exactly one action in step {step_index}, found {len(actions)}")
        return SelectedAction(actions[0].strip(), source="assistant")


class PrecomputedActionSelector:
    def __init__(self, selections: dict[int, SelectedAction]):
        self._selections = selections

    def select_action(self, step_messages: list[dict[str, Any]], *, step_index: int) -> SelectedAction:
        if step_index not in self._selections:
            raise ValueError(f"No precomputed action for step {step_index}")
        return self._selections[step_index]


@dataclass
class ReplayResult:
    history_messages: list[dict[str, Any]]
    replayed_steps: int
    total_action_steps: int
    mismatches: list[dict[str, Any]]
    selected_actions: list[SelectedAction]


class TrajectoryReplayer:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        agent_config: dict[str, Any],
        env: Environment,
        *,
        include_thoughts: bool,
        action_selector: ActionSelector,
        verify_observations: bool = False,
    ) -> None:
        self.messages = messages
        self.steps = messages_to_steps(messages)
        self.agent_config = agent_config
        self.env = env
        self.include_thoughts = include_thoughts
        self.action_selector = action_selector
        self.verify_observations = verify_observations
        self._observation_template = agent_config.get("action_observation_template", "")

    def replay_to_step(self, target_step: int) -> ReplayResult:
        if target_step < 0:
            raise ValueError("target_step must be >= 0")

        action_step_count = 0
        history_steps: list[list[dict[str, Any]]] = []
        mismatches: list[dict[str, Any]] = []
        selected_actions: list[SelectedAction] = []

        for step_index, step in enumerate(self.steps):
            has_assistant = any(message.get("role") == "assistant" for message in step)
            if not has_assistant:
                history_steps.append(step)
                continue

            next_action_index = action_step_count + 1
            if next_action_index > target_step:
                break

            history_steps.append(step)
            action_step_count = next_action_index

            selected = self.action_selector.select_action(step, step_index=step_index)
            output = self.env.execute(selected.action)
            selected_actions.append(
                SelectedAction(
                    action=selected.action,
                    source=selected.source,
                    metadata={
                        **selected.metadata,
                        "returncode": output.get("returncode"),
                        "output_len": len(output.get("output", "") or ""),
                    },
                )
            )

            if self.verify_observations:
                expected = self._expected_observation(step)
                if expected is not None:
                    actual = self._render_observation(output)
                    if actual != expected:
                        mismatches.append(
                            {
                                "step_index": step_index,
                                "expected": expected,
                                "actual": actual,
                            }
                        )

            if action_step_count == target_step:
                break

        total_action_steps = sum(1 for step in self.steps if any(m.get("role") == "assistant" for m in step))
        if target_step > total_action_steps:
            raise ValueError(f"Target step {target_step} exceeds available action steps {total_action_steps}")

        history_messages = build_message_history(
            flatten_steps(history_steps),
            include_thoughts=self.include_thoughts,
        )

        return ReplayResult(
            history_messages=history_messages,
            replayed_steps=target_step,
            total_action_steps=total_action_steps,
            mismatches=mismatches,
            selected_actions=selected_actions,
        )

    def _expected_observation(self, step: list[dict[str, Any]]) -> str | None:
        for message in reversed(step):
            if message.get("role") == "user":
                return _coerce_content(message.get("content"))
        return None

    def _render_observation(self, output: dict[str, Any]) -> str:
        template_vars = self.agent_config | self.env.get_template_vars()
        return Template(self._observation_template, undefined=StrictUndefined).render(output=output, **template_vars)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from minisweagent.agents.default import DefaultAgent


@dataclass
class RolloutActionResult:
    action: str
    output: dict[str, Any]
    source: str
    response_content: str | None
    metadata: dict[str, Any]


class RolloutActionProvider(Protocol):
    def execute(self, agent: DefaultAgent, *, step_index: int) -> RolloutActionResult: ...


class RolloutStepError(Exception):
    def __init__(self, action: str | None, cause: Exception):
        super().__init__(str(cause))
        self.action = action
        self.cause = cause


class ModelActionProvider:
    def execute(self, agent: DefaultAgent, *, step_index: int) -> RolloutActionResult:
        response = agent.query()
        action_dict = agent.parse_action(response)
        action = action_dict["action"]
        try:
            output = agent.execute_action(action_dict)
        except Exception as exc:  # pragma: no cover - handled by caller
            raise RolloutStepError(action, exc) from exc
        observation = agent.render_template(agent.config.action_observation_template, output=output)
        agent.add_message("user", observation)
        return RolloutActionResult(
            action=action,
            output=output,
            source="model",
            response_content=response.get("content"),
            metadata={},
        )


class FixedActionProvider:
    def __init__(self, actions: list[str], *, action_prefix: str = "```bash\n", action_suffix: str = "\n```"):
        if not actions:
            raise ValueError("FixedActionProvider requires at least one action")
        self.actions = actions
        self.action_prefix = action_prefix
        self.action_suffix = action_suffix

    def execute(self, agent: DefaultAgent, *, step_index: int) -> RolloutActionResult:
        if step_index >= len(self.actions):
            raise RolloutStepError(None, ValueError("No fixed action available for step"))
        action = self.actions[step_index]
        content = f"{self.action_prefix}{action}{self.action_suffix}"
        agent.add_message("assistant", content=content)
        try:
            output = agent.get_observation({"content": content})
        except Exception as exc:  # pragma: no cover - handled by caller
            raise RolloutStepError(action, exc) from exc
        return RolloutActionResult(
            action=action,
            output=output,
            source="fixed",
            response_content=content,
            metadata={},
        )

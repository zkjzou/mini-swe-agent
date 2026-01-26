"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import copy
import re
import subprocess
import time
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.run.utils.rejected_actions import RejectedActionSampler
from minisweagent.utils.log import logger


class AgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentConfig,
        rejected_action_sampling: dict | None = None,
        **kwargs,
    ):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._last_prompt_messages: list[dict[str, str]] | None = None
        self._action_step_index = 0
        self._rejected_action_sampling = rejected_action_sampling or {}
        self._rejected_action_sampler: RejectedActionSampler | None = None
        self._init_rejected_action_sampler()

    def _init_rejected_action_sampler(self) -> None:
        cfg = self._rejected_action_sampling
        if not cfg or not cfg.get("enabled"):
            return
        mode = cfg.get("mode", "online")
        if mode not in {"online", "both"}:
            return
        output_path = cfg.get("output_path")
        if not output_path:
            logger.warning("Rejected action sampling enabled but no output_path configured; skipping.")
            return
        model_pool = cfg.get("model_pool", [])
        expert_model_name = cfg.get("expert_model_name") or getattr(self.model.config, "model_name", None)
        expert_model_class = cfg.get("expert_model_class") or (
            f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
        )

        self._rejected_action_sampler = RejectedActionSampler(
            model_pool=model_pool,
            action_regex=cfg.get("action_regex") or self.config.action_regex,
            selection_policy=cfg.get("selection_policy", "round_robin"),
            k_per_step=cfg.get("k_per_step", 1),
            max_attempts_per_step=cfg.get("max_attempts_per_step"),
            timeout_s=cfg.get("timeout_s"),
            expert_model_name=expert_model_name,
            expert_model_class=expert_model_class,
            output_path=Path(output_path),
            overwrite=bool(cfg.get("overwrite", False)),
            run_id=cfg.get("run_id"),
            mode="online",
        )

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        self._last_prompt_messages = copy.deepcopy(self.messages)
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        self._maybe_sample_rejected_actions(action.get("action", ""))
        output = self.execute_action(action)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def _maybe_sample_rejected_actions(self, expert_action: str) -> None:
        if not self._rejected_action_sampler or not self._last_prompt_messages:
            return
        step_index = self._action_step_index
        self._action_step_index += 1
        try:
            self._rejected_action_sampler.sample_and_write(
                prompt_messages=self._last_prompt_messages,
                step_index=step_index,
                expert_action=expert_action,
            )
        except Exception as exc:
            logger.warning("Rejected action sampling failed at step %s: %s", step_index, exc)

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

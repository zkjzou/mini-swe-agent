"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
import time
from typing import Any, Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

from minisweagent import Environment, Model
from minisweagent.models import get_model
from minisweagent.verifiers.first_valid import FirstValidVerifier
from minisweagent.verifiers.llm import LLMVerifier


class CandidateSamplingConfig(BaseModel):
    num_candidates: int = 1
    """How many candidate actions to sample per step."""
    use_n: bool = False
    """If True, try to request multiple candidates in a single model call using n."""
    sampling_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs passed to model.query for sampling (e.g., temperature, top_p, max_tokens)."""


class VerifierConfig(BaseModel):
    enabled: bool = False
    verifier_type: Literal["first_valid", "llm"] = "first_valid"
    """Which verifier to use when enabled."""
    model: dict[str, Any] = Field(default_factory=dict)
    """Model configuration for the verifier (can differ from the agent model)."""
    system_template: str = (
        "You are a verifier that selects the best candidate action for the agent to execute."
    )
    selection_template: str = (
        "Choose the best candidate action for the task. "
        "Return only the number of the chosen candidate.\n\n"
        "{% for c in candidates %}"
        "Candidate {{ c.index + selection_index_base }}:\n"
        "{{ c.content }}\n"
        "{% endfor %}"
    )
    selection_regex: str = r"(\d+)"
    selection_index_base: int = 1
    fallback: Literal["first_candidate", "first_valid"] = "first_candidate"


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
    candidate_sampling: CandidateSamplingConfig = Field(default_factory=CandidateSamplingConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)


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
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.verifier = self._build_verifier()

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
        self._check_limits()
        responses, candidate_infos = self._sample_candidates()
        selected_index = 0
        verifier_metadata = {"enabled": False}
        if self.verifier:
            selected_index, verifier_output = self.verifier.select(
                candidates=candidate_infos,
                template_vars=self.extra_template_vars,
            )
            selected_index = max(0, min(selected_index, len(responses) - 1))
            verifier_metadata = {
                "enabled": True,
                "type": self.config.verifier.verifier_type,
                "selected_index": selected_index,
                "selection_index_base": self.config.verifier.selection_index_base,
                "candidates": candidate_infos,
                "verifier_output": verifier_output,
            }
        response = responses[selected_index]
        if self.verifier:
            response = self._attach_verifier_metadata(response, verifier_metadata)
        self.add_message("assistant", **response)
        return response

    def _check_limits(self) -> None:
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

    def _build_verifier(self):
        if not self.config.verifier.enabled:
            return None
        if self.config.verifier.verifier_type == "first_valid":
            return FirstValidVerifier(self.config.verifier)
        if self.config.verifier.verifier_type == "llm":
            model_config = self.config.verifier.model
            verifier_model = self.model
            if model_config:
                verifier_model = get_model(model_config.get("model_name"), model_config)
            return LLMVerifier(verifier_model, self.config.verifier)
        raise ValueError(f"Unknown verifier type: {self.config.verifier.verifier_type}")

    def _sample_candidates(self) -> tuple[list[dict], list[dict[str, Any]]]:
        sampling_config = self.config.candidate_sampling
        num_candidates = max(1, sampling_config.num_candidates)
        responses: list[dict] = []
        if num_candidates == 1:
            responses = [self._query_once(**sampling_config.sampling_kwargs)]
        else:
            if sampling_config.use_n:
                response = None
                try:
                    response = self._query_once(n=num_candidates, **sampling_config.sampling_kwargs)
                    responses = self._split_n_response(response, num_candidates)
                except (TerminatingException, NonTerminatingException):
                    raise
                except Exception:
                    responses = []
                if not responses and response is not None:
                    responses = [response]
            while len(responses) < num_candidates:
                responses.append(self._query_once(**sampling_config.sampling_kwargs))
        candidate_infos = [self._build_candidate_info(response, idx) for idx, response in enumerate(responses)]
        return responses, candidate_infos

    def _query_once(self, **kwargs) -> dict:
        self._check_limits()
        return self.model.query(self.messages, **kwargs)

    def _split_n_response(self, response: dict, num_candidates: int) -> list[dict]:
        extra = response.get("extra", {}) or {}
        raw = extra.get("response")
        if not isinstance(raw, dict):
            return []
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        responses: list[dict] = []
        for idx, choice in enumerate(choices[:num_candidates]):
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content") or ""
            responses.append(self._make_choice_response(response, content, idx))
        return responses

    def _make_choice_response(self, response: dict, content: str, choice_index: int) -> dict:
        extra = dict(response.get("extra", {}) or {})
        extra["choice_index"] = choice_index
        return {"content": content, "extra": extra}

    def _build_candidate_info(self, response: dict, index: int) -> dict[str, Any]:
        content = response.get("content", "")
        action, actions_found = self._extract_action(content)
        info: dict[str, Any] = {
            "index": index,
            "content": content,
            "action": action,
            "actions_found": actions_found,
        }
        if action is None:
            info["parse_error"] = f"expected 1 action, found {actions_found}"
        return info

    def _extract_action(self, content: str) -> tuple[str | None, int]:
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip(), len(actions)
        return None, len(actions)

    def _attach_verifier_metadata(self, response: dict, metadata: dict) -> dict:
        extra = dict(response.get("extra", {}) or {})
        extra["verifier"] = metadata
        response["extra"] = extra
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

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

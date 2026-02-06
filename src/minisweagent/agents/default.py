"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

from __future__ import annotations

import copy
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded
from minisweagent.models import get_model
from minisweagent.models.utils.actions_text import parse_regex_actions
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.serialize import recursive_merge
from minisweagent.verifiers.first_valid import FirstValidVerifier
from minisweagent.verifiers.llm import LLMVerifier
from minisweagent.verifiers.prompt_loader import apply_prompt_overrides
from minisweagent.verifiers.reward_model import RewardModelVerifier


class CandidateSamplingConfig(BaseModel):
    num_candidates: int = 1
    """How many candidate actions to sample per step."""
    use_n: bool = False
    """If True, try to request multiple candidates in a single model call using n."""
    sampling_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs passed to model.query for sampling."""


class VerifierConfig(BaseModel):
    enabled: bool = False
    verifier_type: Literal["first_valid", "llm", "reward_model"] = "first_valid"
    """Which verifier to use when enabled."""
    model: dict[str, Any] = Field(default_factory=dict)
    """Model configuration for the verifier (can differ from the actor model)."""
    prompt_name: str | None = None
    """Optional prompt alias to load templates from prompts/verifier/<prompt_name>."""
    prompt_dir: str = "prompts/verifier"
    """Directory containing prompt variants (relative to CWD unless absolute)."""
    system_template: str = "You are a verifier that selects the best candidate action for the agent to execute."
    history_steps: int = 6
    """How many recent action+observation steps to pass to the verifier. Use -1 for all steps."""
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
    reward_system_template: str = "You are a reward model that scores candidate actions for a coding agent."
    reward_prompt_template: str = (
        "Score the candidate action for how well it advances the task safely and correctly. "
        "Return a single line: REWARD: <number>.\n\n"
        "Task: {{ task }}\n"
        "Candidate:\n"
        "{{ candidate.content }}\n"
    )
    reward_regex: str = r"REWARD:\s*([+-]?\d+(?:\.\d+)?)"
    fallback: Literal["first_candidate", "first_valid"] = "first_candidate"


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of *executed* steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    add_format_error_to_conversation: bool = True
    """Whether to append format-error feedback messages to the conversation history."""
    candidate_sampling: CandidateSamplingConfig = Field(default_factory=CandidateSamplingConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.step_count = 0
        self.verifier = self._build_verifier()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost, "step_count": self.step_count},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.step_count = 0
        self.cost = 0.0
        self.n_calls = 0
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                if isinstance(e, FormatError) and not self.config.add_format_error_to_conversation:
                    continue
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                if self.config.output_path:
                    self.save(self.config.output_path)
            if self.messages and self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {}) if self.messages else {}

    def step(self) -> list[dict]:
        """Query the LM, execute actions, and count one executed step."""
        observations = self.execute_actions(self.query())
        self.step_count += 1
        return observations

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        self._check_limits()
        responses, candidate_infos = self._sample_candidates()
        verifier_steps = self._get_verifier_steps()
        verifier_messages = [message for step in verifier_steps for message in step]
        verifier_vars = {
            **self.extra_template_vars,
            "task": self.extra_template_vars.get("task", ""),
            "messages": verifier_messages,
            "steps": verifier_steps,
        }

        selected_index = 0
        verifier_output: dict[str, Any] = {}
        verifier_metadata: dict[str, Any] = {
            "enabled": False,
            "type": "none",
            "selected_index": 0,
            "selection_index_base": self.config.verifier.selection_index_base,
            "candidates": candidate_infos,
            "verifier_output": {},
        }

        if self.verifier:
            if self.config.verifier.verifier_type == "reward_model":
                selected_index, verifier_output = self.verifier.select(
                    candidates=candidate_infos,
                    template_vars=verifier_vars,
                    task=verifier_vars["task"],
                    messages=verifier_messages,
                    steps=verifier_steps,
                )
            else:
                selected_index, verifier_output = self.verifier.select(
                    candidates=candidate_infos,
                    template_vars=verifier_vars,
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

        message = copy.deepcopy(responses[selected_index])
        message = self._attach_verifier_metadata(message, verifier_metadata)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _check_limits(self) -> None:
        if 0 < self.config.step_limit <= self.step_count or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

    def _build_verifier(self):
        if not self.config.verifier.enabled:
            return None
        verifier_config = apply_prompt_overrides(self.config.verifier.model_copy(deep=True))
        if verifier_config.verifier_type == "first_valid":
            return FirstValidVerifier(verifier_config)
        verifier_model = self.model
        if verifier_config.model:
            verifier_model = get_model(verifier_config.model.get("model_name"), verifier_config.model)
        if verifier_config.verifier_type == "llm":
            return LLMVerifier(verifier_model, verifier_config)
        if verifier_config.verifier_type == "reward_model":
            return RewardModelVerifier(verifier_model, verifier_config)
        raise ValueError(f"Unknown verifier_type: {verifier_config.verifier_type}")

    def _sample_candidates(self) -> tuple[list[dict], list[dict[str, Any]]]:
        sampling_config = self.config.candidate_sampling
        num_candidates = max(1, sampling_config.num_candidates)
        responses: list[dict] = []

        if num_candidates == 1:
            responses = [self._query_once(**sampling_config.sampling_kwargs)]
        else:
            if sampling_config.use_n:
                batched_response = None
                try:
                    batched_response = self._query_once(n=num_candidates, **sampling_config.sampling_kwargs)
                    responses = self._split_n_response(batched_response, num_candidates)
                except InterruptAgentFlow:
                    raise
                except Exception:
                    self.logger.exception("Error while splitting n-sampled candidates; falling back to repeated calls.")
                    responses = []
                if not responses and batched_response is not None:
                    responses = [batched_response]

            while len(responses) < num_candidates:
                responses.append(self._query_once(**sampling_config.sampling_kwargs))

        candidate_infos = [self._build_candidate_info(response, idx) for idx, response in enumerate(responses)]
        return responses, candidate_infos

    def _query_once(self, **kwargs) -> dict:
        self._check_limits()
        self.n_calls += 1
        message = self.model.query(self.messages, **kwargs)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        return message

    def _split_n_response(self, response: dict, num_candidates: int) -> list[dict]:
        raw_response = (response.get("extra", {}) or {}).get("response")
        if not isinstance(raw_response, dict):
            return []
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        candidates: list[dict] = []
        for idx, choice in enumerate(choices[:num_candidates]):
            choice_message = choice.get("message", {}) if isinstance(choice, dict) else {}
            candidates.append(self._make_choice_response(response, choice_message, idx))
        return candidates

    def _make_choice_response(self, base_response: dict, choice_message: dict, choice_index: int) -> dict:
        response = copy.deepcopy(base_response)
        if isinstance(choice_message, dict):
            if "role" in choice_message:
                response["role"] = choice_message.get("role")
            if "content" in choice_message:
                response["content"] = choice_message.get("content")
            if "tool_calls" in choice_message:
                response["tool_calls"] = choice_message.get("tool_calls")

        parsed_actions, parse_error = self._parse_actions_from_choice(choice_message)
        extra = dict(response.get("extra", {}) or {})
        if not parsed_actions and choice_index == 0:
            parsed_actions = list((base_response.get("extra", {}) or {}).get("actions", []))
        extra["actions"] = parsed_actions
        extra["choice_index"] = choice_index
        if parse_error:
            extra["candidate_parse_error"] = parse_error
        response["extra"] = extra
        return response

    def _parse_actions_from_choice(self, choice_message: dict) -> tuple[list[dict], str | None]:
        if not isinstance(choice_message, dict):
            return [], "Malformed choice message."

        tool_calls = choice_message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return self._parse_tool_call_actions(tool_calls)

        content = choice_message.get("content")
        if isinstance(content, str):
            action_regex = getattr(getattr(self.model, "config", None), "action_regex", "")
            format_error_template = getattr(getattr(self.model, "config", None), "format_error_template", "{{ error }}")
            if action_regex:
                try:
                    return parse_regex_actions(
                        content,
                        action_regex=action_regex,
                        format_error_template=format_error_template,
                    ), None
                except FormatError as e:
                    return [], self._format_interrupt_message(e)
        return [], None

    def _parse_tool_call_actions(self, tool_calls: list[dict]) -> tuple[list[dict], str | None]:
        actions: list[dict] = []
        errors: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                errors.append("Malformed tool call object.")
                continue
            function_data = tool_call.get("function", {}) or {}
            function_name = function_data.get("name")
            raw_arguments = function_data.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except Exception as e:
                errors.append(f"Error parsing tool call arguments: {e}")
                continue

            if function_name != "bash":
                errors.append(f"Unknown tool '{function_name}'.")
                continue
            if not isinstance(arguments, dict) or "command" not in arguments:
                errors.append("Missing 'command' argument in bash tool call.")
                continue

            action = {"command": arguments["command"]}
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                action["tool_call_id"] = tool_call_id
            actions.append(action)

        parse_error = "; ".join(errors) if errors else None
        return actions, parse_error

    def _format_interrupt_message(self, e: InterruptAgentFlow) -> str:
        messages = list(getattr(e, "messages", []))
        if not messages:
            return type(e).__name__
        return get_content_string(messages[0]) or type(e).__name__

    def _build_candidate_info(self, response: dict, index: int) -> dict[str, Any]:
        extra = response.get("extra", {}) or {}
        actions = extra.get("actions", []) or []
        first_action = actions[0]["command"] if actions else None
        candidate_info = {
            "index": index,
            "content": get_content_string(response),
            "action": first_action,
            "actions": actions,
            "n_actions": len(actions),
        }
        if parse_error := extra.get("candidate_parse_error"):
            candidate_info["parse_error"] = parse_error
        return candidate_info

    def _attach_verifier_metadata(self, response: dict, metadata: dict) -> dict:
        extra = dict(response.get("extra", {}) or {})
        extra["verifier"] = metadata
        response["extra"] = extra
        return response

    def _get_verifier_steps(self) -> list[list[dict[str, Any]]]:
        steps: list[list[dict[str, Any]]] = []
        current_step: list[dict[str, Any]] = []
        started = False
        for message in self.messages:
            if self._is_assistant_message(message):
                if current_step:
                    steps.append(current_step)
                current_step = [message]
                started = True
            elif started:
                current_step.append(message)
        if current_step:
            steps.append(current_step)

        history_steps = self.config.verifier.history_steps
        if history_steps is None or history_steps < 0:
            return steps
        return steps[-history_steps:]

    def _is_assistant_message(self, message: dict) -> bool:
        if message.get("role") == "assistant":
            return True
        if message.get("object") == "response":
            return True
        if message.get("type") == "message" and message.get("role") == "assistant":
            return True
        return False

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                    "step_count": self.step_count,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data

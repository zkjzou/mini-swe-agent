import copy
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

_VERIFIER_FEEDBACK_TEMPLATE = (
    "Verifier feedback from the previous step:\n"
    "{% if previous_verifier_feedback.action %}"
    "Executed action: {{ previous_verifier_feedback.action }}\n"
    "{% endif %}"
    "{% if previous_verifier_feedback.score is not none %}"
    "Verifier score: {{ '%.3f'|format(previous_verifier_feedback.score) }}\n"
    "{% endif %}"
    "{% if previous_verifier_feedback.critique %}"
    "Critique: {{ previous_verifier_feedback.critique }}\n"
    "{% endif %}"
    "Use this feedback to inform your next action and avoid repeating the same mistake."
)


class _StaticRewardModel:
    def query(self, messages, **kwargs):
        prompt = messages[-1].get("content", "")
        score = "0.9" if "Option 2" in prompt else "0.2"
        critique = "Prefer the more targeted command." if "Option 2" in prompt else "Too indirect."
        return {"role": "assistant", "content": f"FEEDBACK: {critique}\nREWARD: {score}"}


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_reward_model_selects_highest_reward():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier_feedback_template"] = _VERIFIER_FEEDBACK_TEMPLATE
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REWARD: 0.0", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Option 1", [{"command": "echo first"}]),
            make_output("Option 2", [{"command": "echo second"}]),
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    agent.verifier.model = _StaticRewardModel()
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    assert "Option 2" in response.get("content", "")
    extra = response.get("extra", {})
    verifier = extra.get("verifier", {})
    assert verifier.get("selected_index") == 1
    verifier_output = verifier.get("verifier_output", {})
    rewards = verifier_output.get("rewards")
    assert rewards == [0.2, 0.9]
    assert verifier_output.get("candidate_feedback") == ["Too indirect.", "Prefer the more targeted command."]
    assert verifier_output.get("selected_feedback") == "Prefer the more targeted command."
    assert verifier_output.get("selected_reward") == 0.9
    assert verifier_output.get("api_calls") == 2


def test_reward_model_prompt_only_requests_feedback_when_enabled():
    class _CaptureRewardModel:
        def __init__(self):
            self.prompts: list[str] = []

        def query(self, messages, **kwargs):
            self.prompts.append(messages[-1].get("content", ""))
            return {"role": "assistant", "content": "REWARD: 0.5", "extra": {"cost": 0.0}}

    base_config = _load_default_agent_config()
    base_config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    base_config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
    }

    disabled_agent = DefaultAgent(
        model=DeterministicModel(outputs=[make_output("Option 1", [{"command": "echo first"}])]),
        env=LocalEnvironment(),
        **base_config,
    )
    disabled_capture = _CaptureRewardModel()
    disabled_agent.verifier.model = disabled_capture
    disabled_agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})
    disabled_agent.query()

    enabled_config = copy.deepcopy(base_config)
    enabled_config["enable_verbal_feedback"] = True
    enabled_config["verifier_feedback_template"] = _VERIFIER_FEEDBACK_TEMPLATE
    enabled_agent = DefaultAgent(
        model=DeterministicModel(outputs=[make_output("Option 1", [{"command": "echo first"}])]),
        env=LocalEnvironment(),
        **enabled_config,
    )
    enabled_capture = _CaptureRewardModel()
    enabled_agent.verifier.model = enabled_capture
    enabled_agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})
    enabled_agent.query()

    assert "FEEDBACK:" not in disabled_capture.prompts[-1]
    assert "FEEDBACK:" in enabled_capture.prompts[-1]


def test_reward_model_checklist_mode_attaches_progress_metadata():
    class _ChecklistAwareRewardModel:
        def query(self, messages, **kwargs):
            prompt = messages[-1].get("content", "")
            if "Generate" in prompt and "checklist items" in prompt:
                return {
                    "role": "assistant",
                    "content": "CHECKLIST:\n- Reproduce issue\n- Implement fix\n- Validate tests\n",
                    "extra": {"cost": 0.2},
                }
            if "Option 2" in prompt:
                return {
                    "role": "assistant",
                    "content": (
                        "REASONING: better option\n"
                        "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.8\n- Item 2: 0.7\n- Item 3: 0.4\n"
                        "PROGRESS: 0.7\nFEEDBACK: Continue with the focused fix path.\nREWARD: 0.9"
                    ),
                    "extra": {"cost": 0.3},
                }
            return {
                "role": "assistant",
                "content": (
                    "REASONING: weaker option\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.3\n- Item 2: 0.2\n- Item 3: 0.1\n"
                    "PROGRESS: 0.2\nFEEDBACK: This does not address the main blocker.\nREWARD: 0.2"
                ),
                "extra": {"cost": 0.3},
            }

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["enable_verbal_feedback"] = True
    config["verifier_feedback_template"] = _VERIFIER_FEEDBACK_TEMPLATE
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        "checklist_mode": "issue_progress",
        "checklist_generate_once": True,
        "checklist_min_items": 3,
        "checklist_max_items": 5,
    }

    model = DeterministicModel(
        outputs=[
            make_output("Option 1", [{"command": "echo first"}]),
            make_output("Option 2", [{"command": "echo second"}]),
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    agent.verifier.model = _ChecklistAwareRewardModel()
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    verifier_output = response.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    checklist = verifier_output.get("checklist", {})

    assert verifier_output.get("rewards") == [0.2, 0.9]
    assert verifier_output.get("candidate_progress_scores") == [0.2, 0.7]
    assert verifier_output.get("candidate_feedback") == [
        "This does not address the main blocker.",
        "Continue with the focused fix path.",
    ]
    assert verifier_output.get("selected_feedback") == "Continue with the focused fix path."
    assert verifier_output.get("candidate_checklist_item_scores") == [[0.3, 0.2, 0.1], [0.8, 0.7, 0.4]]
    assert verifier_output.get("api_calls") == 2
    assert checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate tests"]
    assert checklist.get("generated_this_step") is True
    assert checklist.get("api_calls") == 1
    assert agent.verifier_cost == 0.8


def test_dynamic_checklist_modify_mode_uses_previous_checklist_context():
    class _DynamicChecklistRewardModel:
        def __init__(self):
            self.checklist_prompts: list[str] = []

        def query(self, messages, **kwargs):
            prompt = messages[-1].get("content", "")
            if "Output format:" in prompt and "CHECKLIST:" in prompt:
                self.checklist_prompts.append(prompt)
                if len(self.checklist_prompts) == 1:
                    return {
                        "role": "assistant",
                        "content": "CHECKLIST:\n- Reproduce issue\n- Implement fix\n- Validate tests\n",
                        "extra": {"cost": 0.1},
                    }
                return {
                    "role": "assistant",
                    "content": "CHECKLIST:\n- Confirm repro still valid\n- Implement fix\n- Validate tests\n",
                    "extra": {"cost": 0.1},
                }
            return {
                "role": "assistant",
                "content": (
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.4\n- Item 2: 0.5\n- Item 3: 0.6\n"
                    "PROGRESS: 0.4\nFEEDBACK: Keep the checklist aligned with the latest evidence.\nREWARD: 0.5"
                ),
                "extra": {"cost": 0.1},
            }

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        "checklist_mode": "issue_progress",
        "checklist_dynamic": True,
        "checklist_update_mode": "modify",
        "checklist_generate_once": True,
        "checklist_min_items": 3,
        "checklist_max_items": 5,
    }

    model = DeterministicModel(
        outputs=[
            make_output("Option 1", [{"command": "echo first"}]),
            make_output("Option 1 again", [{"command": "echo second"}]),
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    dynamic_model = _DynamicChecklistRewardModel()
    agent.verifier.model = dynamic_model
    assert agent.verifier.config.prompt_name == "dynamic_checklist_modify/reward"
    assert "Existing checklist:" in agent.verifier.config.checklist_prompt_template
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    first = agent.query()
    second = agent.query()

    first_verifier_output = first.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    second_verifier_output = second.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    first_checklist = first_verifier_output.get("checklist", {})
    second_checklist = second_verifier_output.get("checklist", {})

    assert len(dynamic_model.checklist_prompts) == 2
    assert "Existing checklist:" not in dynamic_model.checklist_prompts[0]
    assert "Generate 3 to 5 checklist items" in dynamic_model.checklist_prompts[0]
    assert "Existing checklist:" in dynamic_model.checklist_prompts[1]
    assert "1. Reproduce issue" in dynamic_model.checklist_prompts[1]
    assert first_checklist.get("generated_this_step") is True
    assert second_checklist.get("generated_this_step") is True
    assert first_checklist.get("dynamic") is True
    assert second_checklist.get("dynamic") is True
    assert first_checklist.get("update_mode") == "modify"
    assert second_checklist.get("update_mode") == "modify"
    assert first_checklist.get("source") == "static_checklist_seed"
    assert second_checklist.get("source") == "dynamic_checklist"
    assert first_checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate tests"]
    assert second_checklist.get("items") == ["Confirm repro still valid", "Implement fix", "Validate tests"]
    assert abs(agent.verifier_cost - 0.4) < 1e-9


def test_reward_model_feedback_is_injected_into_next_actor_query():
    class _CaptureActorModel(DeterministicModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seen_queries: list[list[dict]] = []

        def query(self, messages, **kwargs):
            self.seen_queries.append(copy.deepcopy(messages))
            return super().query(messages, **kwargs)

    class _FeedbackAwareRewardModel:
        def query(self, messages, **kwargs):
            prompt = messages[-1].get("content", "")
            if "Option 2" in prompt or "Second round option 2" in prompt:
                return {
                    "role": "assistant",
                    "content": "FEEDBACK: Prefer the targeted command that narrows the search.\nREWARD: 0.9",
                }
            return {
                "role": "assistant",
                "content": "FEEDBACK: This command is too broad for the current blocker.\nREWARD: 0.2",
            }

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["enable_verbal_feedback"] = True
    config["verifier_feedback_template"] = _VERIFIER_FEEDBACK_TEMPLATE
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REWARD: 0.0", [])],
        },
    }

    model = _CaptureActorModel(
        outputs=[
            make_output("Option 1", [{"command": "echo first"}]),
            make_output("Option 2", [{"command": "echo second"}]),
            make_output("Second round option 1", [{"command": "echo round-two-first"}]),
            make_output("Second round option 2", [{"command": "echo round-two-second"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.verifier.model = _FeedbackAwareRewardModel()
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    first = agent.query()
    second = agent.query()

    assert first.get("extra", {}).get("verifier", {}).get("verifier_output", {}).get("selected_feedback") == (
        "Prefer the targeted command that narrows the search."
    )
    assert second.get("extra", {}).get("verifier", {}).get("verifier_output", {}).get("selected_feedback") == (
        "Prefer the targeted command that narrows the search."
    )

    first_round_prompts = model.seen_queries[:2]
    second_round_prompts = model.seen_queries[2:]
    assert first_round_prompts
    assert second_round_prompts
    assert all(
        "Verifier feedback from the previous step:" not in message.get("content", "")
        for prompt in first_round_prompts
        for message in prompt
        if isinstance(message.get("content"), str)
    )
    for prompt in second_round_prompts:
        assert prompt[-1]["role"] == "user"
        assert "Verifier feedback from the previous step:" in prompt[-1]["content"]
        assert "Executed action: echo second" in prompt[-1]["content"]
        assert "Verifier score: 0.900" in prompt[-1]["content"]
        assert "Critique: Prefer the targeted command that narrows the search." in prompt[-1]["content"]


def test_reward_model_multi_turn_history_excludes_feedback_from_verifier_inputs():
    class _CaptureRewardModel:
        def __init__(self):
            self.seen_queries: list[list[dict]] = []

        def query(self, messages, **kwargs):
            self.seen_queries.append(copy.deepcopy(messages))
            prompt = messages[-1].get("content", "")
            reward = "0.9" if "Option 2" in prompt or "Second round option 2" in prompt else "0.2"
            return {"role": "assistant", "content": f"FEEDBACK: Focus the next step.\nREWARD: {reward}"}

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["enable_verbal_feedback"] = True
    config["verifier_feedback_template"] = _VERIFIER_FEEDBACK_TEMPLATE
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        "include_inputs_in_output": True,
        "history_message_format": "multi_turn_chat",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REWARD: 0.0", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Option 1", [{"command": "echo first"}]),
            make_output("Option 2", [{"command": "echo second"}]),
            make_output("Second round option 1", [{"command": "echo round-two-first"}]),
            make_output("Second round option 2", [{"command": "echo round-two-second"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    capture_model = _CaptureRewardModel()
    agent.verifier.model = capture_model
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.query()
    second = agent.query()

    second_round_verifier_inputs = second.get("extra", {}).get("verifier", {}).get("verifier_output", {}).get("inputs", [])
    assert second_round_verifier_inputs
    first_candidate_messages = second_round_verifier_inputs[0]["messages"]
    assert [message["role"] for message in first_candidate_messages] == ["system", "assistant", "user"]
    assert "Verifier feedback from the previous step:" not in "\n".join(
        message["content"] for message in first_candidate_messages if isinstance(message.get("content"), str)
    )
    assert "FEEDBACK:" not in "\n".join(
        message["content"] for message in first_candidate_messages[:-1] if isinstance(message.get("content"), str)
    )
    assert capture_model.seen_queries

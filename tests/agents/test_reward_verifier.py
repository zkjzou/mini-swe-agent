from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


class _StaticRewardModel:
    def query(self, messages, **kwargs):
        prompt = messages[-1].get("content", "")
        score = "0.9" if "Option 2" in prompt else "0.2"
        return {"role": "assistant", "content": f"REWARD: {score}"}


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_reward_model_selects_highest_reward():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
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
    assert verifier_output.get("api_calls") == 2


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
                        "PROGRESS: 0.7\nREWARD: 0.9"
                    ),
                    "extra": {"cost": 0.3},
                }
            return {
                "role": "assistant",
                "content": (
                    "REASONING: weaker option\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.3\n- Item 2: 0.2\n- Item 3: 0.1\n"
                    "PROGRESS: 0.2\nREWARD: 0.2"
                ),
                "extra": {"cost": 0.3},
            }

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
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
                    "PROGRESS: 0.4\nREWARD: 0.5"
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

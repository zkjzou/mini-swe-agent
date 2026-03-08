import json
import tempfile
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


def test_agent_save_includes_class_names():
    """Test that agent.save includes the full class names with import paths."""
    import yaml

    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        default_config = yaml.safe_load(f)["agent"]

    model = DeterministicModel(outputs=[make_output("echo 'test'", [])])
    env = LocalEnvironment()
    agent = DefaultAgent(model, env, **default_config)

    agent.add_messages({"role": "system", "content": "test system message"})
    agent.add_messages({"role": "user", "content": "test user message"})

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_trajectory.json"

        agent.save(temp_path, {"info": {"exit_status": "Submitted", "submission": "test result"}})

        with temp_path.open() as f:
            saved_data = json.load(f)

        assert "info" in saved_data
        assert "config" in saved_data["info"]

        config = saved_data["info"]["config"]

        assert "agent_type" in config
        assert "model_type" in config
        assert "environment_type" in config

        assert config["agent_type"] == "minisweagent.agents.default.DefaultAgent"
        assert config["model_type"] == "minisweagent.models.test_models.DeterministicModel"
        assert config["environment_type"] == "minisweagent.environments.local.LocalEnvironment"

        assert saved_data["info"]["exit_status"] == "Submitted"
        assert saved_data["info"]["submission"] == "test result"
        assert saved_data["trajectory_format"] == "mini-swe-agent-1.1"


def test_agent_serialize():
    """Test that agent.serialize returns the correct structure."""
    import yaml

    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        default_config = yaml.safe_load(f)["agent"]

    model = DeterministicModel(outputs=[make_output("echo 'test'", [])])
    env = LocalEnvironment()
    agent = DefaultAgent(model, env, **default_config)

    agent.add_messages({"role": "system", "content": "test system message"})
    agent.add_messages({"role": "user", "content": "test user message"})

    data = agent.serialize()

    assert "info" in data
    assert "config" in data["info"]
    assert "messages" in data


def test_agent_save_includes_verifier_feedback_message_in_messages():
    class _StaticRewardModel:
        def query(self, messages, **kwargs):
            prompt = messages[-1].get("content", "")
            score = "0.9" if "Option 2" in prompt else "0.2"
            critique = "Prefer the more targeted command." if "Option 2" in prompt else "Too indirect."
            return {"role": "assistant", "content": f"FEEDBACK: {critique}\nREWARD: {score}"}

    config_path = Path("src/minisweagent/config/default.yaml")
    default_config = yaml.safe_load(config_path.read_text())["agent"]
    default_config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    default_config["enable_verbal_feedback"] = True
    default_config["verifier_feedback_template"] = (
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
        "Treat this as advisory feedback, not ground truth."
    )
    default_config["verifier"] = {
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
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **default_config)
    agent.verifier.model = _StaticRewardModel()
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})
    agent.query()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_feedback_trajectory.json"
        agent.save(temp_path)
        saved_data = json.loads(temp_path.read_text())

    assistant_messages = [message for message in saved_data["messages"] if message.get("role") == "assistant"]
    assert assistant_messages
    verifier = assistant_messages[-1].get("extra", {}).get("verifier", {})
    assert (
        verifier.get("feedback_message")
        == "Verifier feedback from the previous step:\n"
        "Executed action: echo second\n"
        "Verifier score: 0.900\n"
        "Critique: Prefer the more targeted command.\n"
        "Treat this as advisory feedback, not ground truth."
    )

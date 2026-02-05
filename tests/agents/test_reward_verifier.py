from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_reward_model_selects_highest_reward():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": ["REWARD: 0.2", "REWARD: 0.9"],
        },
        "reward_regex": r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
    }

    model = DeterministicModel(
        outputs=[
            "Option 1\n```bash\necho 'first'\n```",
            "Option 2\n```bash\necho 'second'\n```",
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    agent.add_message("system", "system")
    agent.add_message("user", "task")

    response = agent.query()
    assert "second" in response.get("content", "")
    extra = response.get("extra", {})
    verifier = extra.get("verifier", {})
    assert verifier.get("selected_index") == 1
    rewards = verifier.get("verifier_output", {}).get("rewards")
    assert rewards == [0.2, 0.9]

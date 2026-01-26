import json
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.utils.save import save_traj


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_first_valid_verifier_records_metadata(tmp_path):
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {"enabled": True, "verifier_type": "first_valid"}

    model = DeterministicModel(
        outputs=[
            "No action here.",
            "Run hello\n```bash\necho 'hello'\n```",
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    agent.add_message("system", "system")
    agent.add_message("user", "task")

    agent.step()

    assistant_messages = [msg for msg in agent.messages if msg.get("role") == "assistant"]
    assert assistant_messages
    extra = assistant_messages[-1].get("extra", {})
    assert "verifier" in extra
    verifier = extra["verifier"]
    assert verifier["selected_index"] == 1
    assert verifier["candidates"][0]["action"] is None
    assert verifier["candidates"][1]["action"] == "echo 'hello'"

    traj_path = tmp_path / "traj.json"
    save_traj(agent, traj_path, print_path=False)
    data = json.loads(traj_path.read_text())
    saved_messages = data.get("messages", [])
    saved_with_verifier = []
    for msg in saved_messages:
        if msg.get("role") != "assistant":
            continue
        extra = msg.get("extra") or {}
        if "verifier" in extra:
            saved_with_verifier.append(msg)
    assert saved_with_verifier

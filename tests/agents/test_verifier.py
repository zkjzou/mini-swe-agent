import json
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.test_models import DeterministicModel, make_output


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_first_valid_verifier_records_metadata(tmp_path):
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {"enabled": True, "verifier_type": "first_valid"}

    model = DeterministicModel(
        outputs=[
            make_output("No action here.", []),
            make_output("Run hello", [{"command": "echo 'hello'"}]),
        ]
    )
    env = LocalEnvironment()
    agent = DefaultAgent(model=model, env=env, **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.step()

    assistant_messages = [msg for msg in agent.messages if msg.get("role") == "assistant"]
    assert assistant_messages
    extra = assistant_messages[-1].get("extra", {})
    assert "verifier" in extra
    verifier = extra["verifier"]
    assert verifier["selected_index"] == 1
    assert verifier["candidates"][0]["action"] is None
    assert verifier["candidates"][1]["action"] == "echo 'hello'"
    assert verifier["candidates"][1]["actions"] == [{"command": "echo 'hello'"}]

    traj_path = tmp_path / "traj.json"
    agent.save(traj_path)
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
    assert saved_with_verifier[-1]["extra"]["verifier"]["candidates"][1]["action"] == "echo 'hello'"
    assert data["info"]["model_stats"]["step_count"] == 1


def test_llm_verifier_uses_last_index_match():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "selection_regex": r"(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("Reasoning says 1 first.\nFINAL: 2", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "echo first"}]),
            make_output("Candidate 2", [{"command": "echo second"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    verifier = response.get("extra", {}).get("verifier", {})
    assert verifier.get("selected_index") == 1
    assert verifier.get("verifier_output", {}).get("raw_index") == 2
    assert "echo second" in verifier["candidates"][1]["action"]


def test_step_limit_uses_step_count_not_model_calls():
    config = _load_default_agent_config()
    config["step_limit"] = 1
    config["candidate_sampling"] = {"num_candidates": 3, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {"enabled": False}

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "echo first"}]),
            make_output("Candidate 2", [{"command": "echo second"}]),
            make_output("Candidate 3", [{"command": "echo third"}]),
            make_output("Should never be queried", [{"command": "echo never"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.step()
    assert agent.step_count == 1
    assert agent.n_calls == 3

    with pytest.raises(LimitsExceeded):
        agent.step()
    assert agent.n_calls == 3

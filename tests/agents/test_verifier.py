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


def test_similarity_gate_skips_verifier_and_random_samples():
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
            "outputs": [make_output("FINAL: 2", [])],
        },
        "skip_if_actions_similar": True,
        "action_similarity_metric": "token_jaccard",
        "action_similarity_threshold": 0.9,
        "action_similarity_seed": 7,
        "checklist_mode": "issue_progress",
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "python -m pytest -q"}]),
            make_output("Candidate 2", [{"command": "python -m pytest -q"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    verifier = response.get("extra", {}).get("verifier", {})
    verifier_output = verifier.get("verifier_output", {})
    assert verifier.get("type") == "similarity_gate"
    assert verifier_output.get("skipped") is True
    assert verifier_output.get("should_skip_verifier") is True
    assert verifier_output.get("metric") == "token_jaccard"
    assert verifier.get("selected_index") in [0, 1]
    assert "checklist" not in verifier_output
    assert agent.verifier_cost == 0.0
    assert agent.verifier.model.current_index == -1


def test_checklist_mode_generates_once_and_reuses_across_queries():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [
                make_output("CHECKLIST:\n- Reproduce issue\n- Implement fix\n- Validate behavior", []),
                make_output(
                    "REASONING: choose 1\n"
                    "SCORES:\n- Candidate 1: 0.9\n- Candidate 2: 0.2\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.6\n- Item 2: 0.2\n- Item 3: 0.1\n"
                    "PROGRESS: 0.3\nFINAL: 1",
                    [],
                ),
                make_output(
                    "REASONING: choose 2\n"
                    "SCORES:\n- Candidate 1: 0.4\n- Candidate 2: 0.8\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.8\n- Item 2: 0.7\n- Item 3: 0.4\n"
                    "PROGRESS: 0.6\nFINAL: 2",
                    [],
                ),
            ],
        },
        "checklist_mode": "issue_progress",
        "checklist_generate_once": True,
        "checklist_min_items": 3,
        "checklist_max_items": 5,
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1A", [{"command": "echo first-a"}]),
            make_output("Candidate 2A", [{"command": "echo second-a"}]),
            make_output("Candidate 1B", [{"command": "echo first-b"}]),
            make_output("Candidate 2B", [{"command": "echo second-b"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    first = agent.query()
    second = agent.query()

    first_verifier_output = first.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    second_verifier_output = second.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    first_checklist = first_verifier_output.get("checklist", {})
    second_checklist = second_verifier_output.get("checklist", {})

    assert first_checklist.get("generated_this_step") is True
    assert second_checklist.get("generated_this_step") is False
    assert first_checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate behavior"]
    assert second_checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate behavior"]
    assert first_verifier_output.get("progress_score") == 0.3
    assert second_verifier_output.get("progress_score") == 0.6
    assert first_verifier_output.get("checklist_item_scores") == [0.6, 0.2, 0.1]
    assert second_verifier_output.get("checklist_item_scores") == [0.8, 0.7, 0.4]
    assert agent.verifier_cost == 3.0

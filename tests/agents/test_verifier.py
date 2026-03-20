import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicToolcallModel,
    make_output,
    make_toolcall_output,
)


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


def test_verifier_model_defaults_to_litellm_textbased_when_model_class_is_missing():
    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "model": {
            "model_name": "openai/gpt-5-mini",
            "model_kwargs": {"drop_params": True},
        },
    }

    verifier_model = DeterministicModel(outputs=[make_output("FINAL: 1", [])])
    with patch("minisweagent.agents.default.get_model") as mock_get_model:
        mock_get_model.return_value = verifier_model
        agent = DefaultAgent(
            model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
            env=LocalEnvironment(),
            **config,
        )

    called_model_config = mock_get_model.call_args.args[1]
    assert agent.verifier.config.model["model_class"] == "litellm_textbased"
    assert called_model_config["model_class"] == "litellm_textbased"
    assert agent.verifier.model is verifier_model


@pytest.mark.parametrize(
    ("requested_model_class", "expected_model_class", "model_name"),
    [
        ("litellm", "litellm_textbased", "openai/gpt-5-mini"),
        ("openrouter", "openrouter_textbased", "openrouter/openai/gpt-4o-mini"),
    ],
)
def test_verifier_model_rewrites_toolcalling_aliases_to_textbased(
    requested_model_class, expected_model_class, model_name
):
    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "model": {
            "model_class": requested_model_class,
            "model_name": model_name,
            "model_kwargs": {"drop_params": True},
        },
    }

    verifier_model = DeterministicModel(outputs=[make_output("FINAL: 1", [])])
    with patch("minisweagent.agents.default.get_model") as mock_get_model:
        mock_get_model.return_value = verifier_model
        agent = DefaultAgent(
            model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
            env=LocalEnvironment(),
            **config,
        )

    called_model_config = mock_get_model.call_args.args[1]
    assert agent.verifier.config.model["model_class"] == expected_model_class
    assert called_model_config["model_class"] == expected_model_class
    assert agent.verifier.model is verifier_model


def test_verifier_rejects_unsupported_toolcalling_model_classes():
    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "model": {
            "model_class": "litellm_response",
            "model_name": "openai/gpt-5-mini",
            "model_kwargs": {"drop_params": True},
        },
    }

    with pytest.raises(ValueError, match="Verifier model_class 'litellm_response' is not supported"):
        DefaultAgent(
            model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
            env=LocalEnvironment(),
            **config,
        )


def test_verifier_fallback_rejects_non_textbased_actor_model():
    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
    }

    actor_model = DeterministicToolcallModel(outputs=[make_toolcall_output("THOUGHTS: hi", [], [])])
    with pytest.raises(ValueError, match="Verifier is enabled without agent.verifier.model"):
        DefaultAgent(model=actor_model, env=LocalEnvironment(), **config)


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


def test_every_n_steps_runs_verifier_on_step_one_then_cadence():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "verifier_run_policy": "every_n_steps",
        "verifier_run_every_n_steps": 2,
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [
                make_output("REASONING: pick 2\nFINAL: 2", []),
                make_output("REASONING: pick 1\nFINAL: 1", []),
            ],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Step 1 candidate 1", [{"command": "echo step-1-a"}]),
            make_output("Step 1 candidate 2", [{"command": "echo step-1-b"}]),
            make_output("Step 2 candidate 1", [{"command": "echo step-2-a"}]),
            make_output("Step 3 candidate 1", [{"command": "echo step-3-a"}]),
            make_output("Step 3 candidate 2", [{"command": "echo step-3-b"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.step()
    agent.step()
    agent.step()

    assistant_messages = [msg for msg in agent.messages if msg.get("role") == "assistant"]
    assert len(assistant_messages) == 3

    first_verifier = assistant_messages[0]["extra"]["verifier"]
    second_verifier = assistant_messages[1]["extra"]["verifier"]
    third_verifier = assistant_messages[2]["extra"]["verifier"]

    assert first_verifier["type"] == "llm"
    assert first_verifier["selected_index"] == 1
    assert second_verifier["type"] == "schedule_gate"
    assert second_verifier["verifier_output"]["skip_reason"] == "frequency_gate"
    assert second_verifier["verifier_output"]["sampled_candidate_count"] == 1
    assert third_verifier["type"] == "llm"
    assert third_verifier["selected_index"] == 0
    assert agent.n_calls == 5
    assert agent.verifier.model.current_index == 1


def test_editing_command_policy_skips_read_only_commands_without_extra_sampling():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "verifier_run_policy": "editing_commands_only",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REASONING: unreachable\nFINAL: 1", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "rg verifier src/minisweagent"}]),
            make_output("Candidate 2", [{"command": "echo should-not-be-sampled"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()

    verifier = response["extra"]["verifier"]
    assert verifier["type"] == "schedule_gate"
    assert verifier["verifier_output"]["skip_reason"] == "non_editing_action"
    assert verifier["verifier_output"]["candidate_action"] == "rg verifier src/minisweagent"
    assert verifier["verifier_output"]["sampled_candidate_count"] == 1
    assert agent.n_calls == 1
    assert agent.verifier.model.current_index == -1


def test_editing_command_policy_expands_candidates_for_direct_file_edits():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "verifier_run_policy": "editing_commands_only",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REASONING: prefer second\nFINAL: 2", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "sed -i 's/old/new/' src/app.py"}]),
            make_output("Candidate 2", [{"command": "echo validate"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()

    verifier = response["extra"]["verifier"]
    assert verifier["type"] == "llm"
    assert verifier["selected_index"] == 1
    assert len(verifier["candidates"]) == 2
    assert agent.n_calls == 2
    assert agent.verifier.model.current_index == 0


def test_editing_command_policy_treats_python_scripts_as_editing_commands():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "verifier_run_policy": "editing_commands_only",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REASONING: pick 1\nFINAL: 1", [])],
        },
    }

    model = DeterministicModel(
        outputs=[
            make_output("Candidate 1", [{"command": "python scripts/rewrite_verifier.py"}]),
            make_output("Candidate 2", [{"command": "rg verifier src/minisweagent"}]),
        ]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()

    verifier = response["extra"]["verifier"]
    assert verifier["type"] == "llm"
    assert len(verifier["candidates"]) == 2
    assert agent.n_calls == 2
    assert agent.verifier.model.current_index == 0


def test_verifier_rejects_non_positive_every_n_steps():
    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "verifier_run_policy": "every_n_steps",
        "verifier_run_every_n_steps": 0,
    }

    with pytest.raises(ValidationError):
        DefaultAgent(
            model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
            env=LocalEnvironment(),
            **config,
        )


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
    assert first_verifier_output.get("api_calls") == 1
    assert second_verifier_output.get("api_calls") == 1
    assert first_checklist.get("api_calls") == 1
    assert second_checklist.get("api_calls") == 0
    assert agent.verifier_cost == 3.0
    stats = agent.serialize()["info"]["model_stats"]
    assert stats["agent_api_calls"] == 4
    assert stats["verifier_api_calls"] == 2
    assert stats["checklist_api_calls"] == 1
    assert stats["api_calls"] == 7


def test_checklist_prompt_name_enables_checklist_mode_automatically():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 2, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "prompt_name": "checklist/verifier",
        "prompt_dir": "prompts/verifier",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [
                make_output("CHECKLIST:\n- Reproduce issue\n- Implement fix\n- Validate behavior", []),
                make_output(
                    "REASONING: choose 1\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.8\n- REASONING: helpful\n- Item 2: 0.4\n- REASONING: partial\n- Item 3: 0.2\n- REASONING: weak\n"
                    "PROGRESS: Yes + useful\n"
                    "SCORE: 0.9\nFINAL: 1",
                    [],
                ),
            ],
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

    verifier_output = response.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    checklist = verifier_output.get("checklist", {})
    assert checklist.get("generated_this_step") is True
    assert checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate behavior"]


def test_dynamic_checklist_regenerate_mode_refreshes_each_query():
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
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.6\n- Item 2: 0.2\n- Item 3: 0.1\n"
                    "PROGRESS: 0.3\nFINAL: 1",
                    [],
                ),
                make_output("CHECKLIST:\n- Reproduce issue again\n- Patch source\n- Re-run focused tests", []),
                make_output(
                    "REASONING: choose 2\n"
                    "CHECKLIST_ITEM_SCORES:\n- Item 1: 0.7\n- Item 2: 0.8\n- Item 3: 0.6\n"
                    "PROGRESS: 0.6\nFINAL: 2",
                    [],
                ),
            ],
        },
        "checklist_mode": "issue_progress",
        "checklist_dynamic": True,
        "checklist_update_mode": "regenerate",
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
    assert agent.verifier.config.prompt_name == "dynamic_checklist_regenerate/verifier"
    assert "Generate a complete NEW checklist" not in agent.verifier.config.checklist_prompt_template
    assert "Existing checklist:" not in agent.verifier.config.checklist_prompt_template
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    first = agent.query()
    second = agent.query()

    first_verifier_output = first.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    second_verifier_output = second.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    first_checklist = first_verifier_output.get("checklist", {})
    second_checklist = second_verifier_output.get("checklist", {})

    assert first_checklist.get("generated_this_step") is True
    assert second_checklist.get("generated_this_step") is True
    assert first_checklist.get("dynamic") is True
    assert second_checklist.get("dynamic") is True
    assert first_checklist.get("update_mode") == "regenerate"
    assert second_checklist.get("update_mode") == "regenerate"
    assert first_checklist.get("items") == ["Reproduce issue", "Implement fix", "Validate behavior"]
    assert second_checklist.get("items") == ["Reproduce issue again", "Patch source", "Re-run focused tests"]
    assert agent.verifier_cost == 4.0


def test_checklist_v2_omits_min_max_and_parses_rubric():
    class _ChecklistV2VerifierModel:
        def __init__(self):
            self.checklist_prompt: str = ""
            self.call_count = 0

        def query(self, messages, **kwargs):
            self.call_count += 1
            prompt = messages[-1].get("content", "")
            if self.call_count == 1:
                self.checklist_prompt = prompt
                return {
                    "role": "assistant",
                    "content": (
                        "rubric:\n"
                        "  - id: S1\n"
                        "    stage: localization\n"
                        "    weight: 3\n"
                        "    description: Locates failing path in src/minisweagent/agents/default.py\n"
                        "    observable_signal: Stack trace points to checklist generation\n"
                        "  - id: S2\n"
                        "    stage: verification\n"
                        "    weight: 2\n"
                        "    description: Confirms verifier outputs checklist scores for each candidate\n"
                        "    observable_signal: Trajectory includes checklist_item_scores\n"
                    ),
                    "extra": {"cost": 0.1},
                }
            return {
                "role": "assistant",
                "content": (
                    "REASONING: candidate 1 is safer.\n"
                    "CHECKLIST_ITEM_SCORES:\n"
                    "- Item 1: 0.8\n"
                    "- Item 2: 0.6\n"
                    "PROGRESS: 0.7\n"
                    "FINAL: 1"
                ),
                "extra": {"cost": 0.2},
            }

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "prompt_name": "checklist_v2/verifier",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "checklist_mode": "issue_progress",
    }

    model = DeterministicModel(outputs=[make_output("Candidate 1", [{"command": "echo hi"}])])
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    verifier_model = _ChecklistV2VerifierModel()
    agent.verifier.model = verifier_model
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    verifier_output = response.get("extra", {}).get("verifier", {}).get("verifier_output", {})
    checklist = verifier_output.get("checklist", {})

    assert "Generate 3 to 8 checklist items" not in verifier_model.checklist_prompt
    assert checklist.get("checklist_output_format") == "rubric_yaml"
    assert checklist.get("items") == [
        "Locates failing path in src/minisweagent/agents/default.py",
        "Confirms verifier outputs checklist scores for each candidate",
    ]
    assert checklist.get("rubric_items", [])[0]["id"] == "S1"
    assert verifier_output.get("checklist_item_scores") == [0.8, 0.6]


def test_serialize_uses_resolved_llm_prompt_and_blanks_unused_prompt_sections(tmp_path):
    prompt_root = tmp_path / "prompts" / "verifier" / "basic" / "verifier"
    prompt_root.mkdir(parents=True)
    prompt_root.joinpath("system.jinja").write_text("LLM verifier system prompt")
    prompt_root.joinpath("selection.jinja").write_text("LLM verifier selection prompt")

    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "prompt_dir": str(tmp_path / "prompts" / "verifier"),
        "prompt_name": "basic/verifier",
        "checklist_mode": "off",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("FINAL: 1", [])],
        },
    }

    agent = DefaultAgent(
        model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
        env=LocalEnvironment(),
        **config,
    )
    verifier_config = agent.serialize()["info"]["config"]["agent"]["verifier"]

    assert verifier_config["system_template"] == "LLM verifier system prompt"
    assert verifier_config["selection_template"] == "LLM verifier selection prompt"
    assert verifier_config["reward_system_template"] == ""
    assert verifier_config["reward_prompt_template"] == ""
    assert verifier_config["checklist_system_template"] == ""
    assert verifier_config["checklist_prompt_template"] == ""


def test_serialize_uses_resolved_reward_prompt_and_blanks_unused_prompt_sections(tmp_path):
    prompt_root = tmp_path / "prompts" / "verifier" / "domain" / "reward"
    prompt_root.mkdir(parents=True)
    prompt_root.joinpath("system.jinja").write_text("Reward verifier system prompt")
    prompt_root.joinpath("reward.jinja").write_text("Reward verifier scoring prompt")

    config = _load_default_agent_config()
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "reward_model",
        "prompt_dir": str(tmp_path / "prompts" / "verifier"),
        "prompt_name": "domain/reward",
        "checklist_mode": "off",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("FINAL: 0.9", [])],
        },
    }

    agent = DefaultAgent(
        model=DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])]),
        env=LocalEnvironment(),
        **config,
    )
    verifier_config = agent.serialize()["info"]["config"]["agent"]["verifier"]

    assert verifier_config["system_template"] == ""
    assert verifier_config["selection_template"] == ""
    assert verifier_config["reward_system_template"] == "Reward verifier system prompt"
    assert verifier_config["reward_prompt_template"] == "Reward verifier scoring prompt"
    assert verifier_config["checklist_system_template"] == ""
    assert verifier_config["checklist_prompt_template"] == ""


@pytest.mark.parametrize(
    ("history_steps", "expected_visible_steps"),
    [
        (1, 1),
        (-1, 2),
    ],
)
def test_prompt_templates_can_use_history_steps(history_steps, expected_visible_steps):
    class _CaptureVerifierModel:
        def __init__(self):
            self.messages = None

        def query(self, messages, **kwargs):
            self.messages = messages
            return {"role": "assistant", "content": "FINAL: 1", "extra": {"cost": 0.0}}

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "history_steps": history_steps,
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "system_template": "Verifier system",
        "selection_template": (
            "History steps value: {{ history_steps }}\n"
            "{% if history_steps == -1 %}{% set visible_steps = all_steps %}{% else %}"
            "{% set visible_steps = steps %}{% endif %}\n"
            "Visible steps count: {{ visible_steps|length }}\n"
            "{% for c in candidates %}Candidate {{ c.index + selection_index_base }}:\n{{ c.content }}\n{% endfor %}"
        ),
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("FINAL: 1", [])],
        },
    }

    model = DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])])
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    capture_model = _CaptureVerifierModel()
    agent.verifier.model = capture_model
    agent.add_messages(
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "step1"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "step2"},
        {"role": "user", "content": "obs2"},
    )

    agent.query()
    assert capture_model.messages is not None
    prompt = capture_model.messages[-1]["content"]
    assert f"History steps value: {history_steps}" in prompt
    assert f"Visible steps count: {expected_visible_steps}" in prompt


@pytest.mark.parametrize("include_thoughts_in_history_steps", [True, False])
def test_verifier_history_can_optionally_exclude_assistant_content(include_thoughts_in_history_steps):
    class _CaptureVerifierModel:
        def __init__(self):
            self.messages = None

        def query(self, messages, **kwargs):
            self.messages = messages
            return {"role": "assistant", "content": "FINAL: 1", "extra": {"cost": 0.0}}

    secret_thought = "THOUGHT: verifier-super-secret-thought"
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "history_steps": -1,
        "include_thoughts_in_history_steps": include_thoughts_in_history_steps,
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "system_template": "Verifier system",
        "selection_template": (
            "Steps:\n{% for step in steps %}{% for msg in step %}{{ msg.role }}={{ msg.content }}\n{% endfor %}{% endfor %}\n"
            "AllSteps:\n{% for step in all_steps %}{% for msg in step %}{{ msg.role }}={{ msg.content }}\n{% endfor %}{% endfor %}\n"
            "Messages:\n{% for msg in messages %}{{ msg.role }}={{ msg.content }}\n{% endfor %}\n"
            "AllMessages:\n{% for msg in all_messages %}{{ msg.role }}={{ msg.content }}\n{% endfor %}\n"
            "{% for c in candidates %}Candidate {{ c.index + selection_index_base }}:\n{{ c.content }}\n{% endfor %}"
        ),
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("FINAL: 1", [])],
        },
    }

    model = DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])])
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    capture_model = _CaptureVerifierModel()
    agent.verifier.model = capture_model
    agent.add_messages(
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": secret_thought},
        {"role": "user", "content": "obs1"},
    )

    agent.query()
    assert capture_model.messages is not None
    prompt = capture_model.messages[-1]["content"]
    assert "user=obs1" in prompt
    if include_thoughts_in_history_steps:
        assert secret_thought in prompt
    else:
        assert secret_thought not in prompt
        assert "assistant=" in prompt


def test_verifier_candidate_content_strips_think_blocks():
    class _CaptureVerifierModel:
        def __init__(self):
            self.messages = None

        def query(self, messages, **kwargs):
            self.messages = messages
            return {"role": "assistant", "content": "FINAL: 1", "extra": {"cost": 0.0}}

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "history_steps": -1,
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "system_template": "Verifier system",
        "selection_template": "{% for c in candidates %}Candidate {{ c.index + selection_index_base }}:\n{{ c.content }}\n{% endfor %}",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [{"content": "FINAL: 1", "tool_calls": []}],
        },
    }

    model = DeterministicModel(
        outputs=[make_output("<think>internal reasoning</think>\n\nTHOUGHT: keep this\n\n```bash\necho hi\n```", [{"command": "echo hi"}])]
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    capture_model = _CaptureVerifierModel()
    agent.verifier.model = capture_model
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.query()
    assert capture_model.messages is not None
    prompt = capture_model.messages[-1]["content"]
    assert "THOUGHT: keep this" in prompt
    assert "internal reasoning" not in prompt
    assert "<think>" not in prompt


def test_verifier_uses_task_in_system_prompt_and_excludes_actor_system_message():
    class _CaptureVerifierModel:
        def __init__(self):
            self.messages = None

        def query(self, messages, **kwargs):
            self.messages = messages
            return {"role": "assistant", "content": "FINAL: 1", "extra": {"cost": 0.0}}

    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "history_steps": -1,
        "include_thoughts_in_history_steps": True,
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "system_template": "Verifier system task={{ task }}",
        "selection_template": (
            "Messages:\n{% for msg in messages %}{{ msg.role }}={{ msg.content }}\n{% endfor %}\n"
            "{% for c in candidates %}Candidate {{ c.index + selection_index_base }}:\n{{ c.content }}\n{% endfor %}"
        ),
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("FINAL: 1", [])],
        },
    }

    model = DeterministicModel(outputs=[make_output("Candidate", [{"command": "echo hi"}])])
    agent = DefaultAgent(model=model, env=LocalEnvironment(), **config)
    capture_model = _CaptureVerifierModel()
    agent.verifier.model = capture_model
    agent.add_messages(
        {"role": "system", "content": "coding agent system prompt"},
        {"role": "user", "content": "Fix PR 123 parser issue"},
        {"role": "assistant", "content": "inspect files"},
        {"role": "user", "content": "obs1"},
    )
    agent.extra_template_vars["task"] = "Fix PR 123 parser issue"

    agent.query()

    assert capture_model.messages is not None
    assert capture_model.messages[0]["content"] == "Verifier system task=Fix PR 123 parser issue"
    assert "coding agent system prompt" not in capture_model.messages[-1]["content"]


def test_pair_thoughts_with_toolcalls_builds_verifier_candidates_from_one_response():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {
        "num_candidates": 3,
        "use_n": False,
        "pair_thoughts_with_toolcalls": True,
        "sampling_kwargs": {},
    }
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REASONING: pick 2\nFINAL: 2", [])],
        },
    }

    thought_content = (
        "THOUGHTS: Inspect the repository layout.\n\n"
        "THOUGHTS: Reproduce the failure with a focused command.\n\n"
        "THOUGHTS: Read implementation file around the failing path."
    )
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "pytest -q tests/test_x.py"}'},
        },
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "sed -n \\"1,200p\\" src/app.py"}'},
        },
    ]
    actions = [
        {"command": "ls -la", "tool_call_id": "call_1"},
        {"command": "pytest -q tests/test_x.py", "tool_call_id": "call_2"},
        {"command": 'sed -n "1,200p" src/app.py', "tool_call_id": "call_3"},
    ]

    actor_model = DeterministicToolcallModel(outputs=[make_toolcall_output(thought_content, tool_calls, actions)])
    agent = DefaultAgent(model=actor_model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    response = agent.query()
    verifier = response.get("extra", {}).get("verifier", {})
    candidates = verifier.get("candidates", [])

    assert agent.n_calls == 1
    assert verifier.get("selected_index") == 1
    assert len(candidates) == 3
    assert "Inspect the repository layout" in candidates[0]["content"]
    assert "Reproduce the failure" in candidates[1]["content"]
    assert candidates[0]["action"] == "ls -la"
    assert candidates[1]["action"] == "pytest -q tests/test_x.py"
    assert candidates[2]["action"] == 'sed -n "1,200p" src/app.py'
    assert response.get("extra", {}).get("actions", [{}])[0].get("command") == "pytest -q tests/test_x.py"


def test_pair_thoughts_with_toolcalls_raises_format_error_on_count_mismatch():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {
        "num_candidates": 2,
        "use_n": False,
        "pair_thoughts_with_toolcalls": True,
        "sampling_kwargs": {},
    }
    config["verifier"] = {"enabled": False}

    content = "THOUGHTS: only one thought block."
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
        },
    ]
    actions = [
        {"command": "ls -la", "tool_call_id": "call_1"},
        {"command": "pwd", "tool_call_id": "call_2"},
    ]
    actor_model = DeterministicToolcallModel(outputs=[make_toolcall_output(content, tool_calls, actions)])
    agent = DefaultAgent(model=actor_model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    with pytest.raises(FormatError) as exc_info:
        agent.query()
    format_msg = (exc_info.value.messages or [{}])[0].get("content", "")
    assert "matching counts of THOUGHTS sections and tool calls" in format_msg


def test_llm_verifier_multi_turn_history_includes_tool_calls_and_outputs():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}}
    config["verifier"] = {
        "enabled": True,
        "verifier_type": "llm",
        "selection_regex": r"FINAL:\s*(\d+)",
        "selection_index_base": 1,
        "include_inputs_in_output": True,
        "history_message_format": "multi_turn_chat",
        "model": {
            "model_class": "deterministic",
            "model_name": "deterministic",
            "outputs": [make_output("REASONING: pick 1\nFINAL: 1", []), make_output("REASONING: pick 1\nFINAL: 1", [])],
        },
    }

    first_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
        }
    ]
    second_tool_calls = [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo second"}'},
        }
    ]
    actor_model = DeterministicToolcallModel(
        outputs=[
            make_toolcall_output("Inspect current state", first_tool_calls, [{"command": "echo hello", "tool_call_id": "call_1"}]),
            make_toolcall_output("Continue with next action", second_tool_calls, [{"command": "echo second", "tool_call_id": "call_2"}]),
        ]
    )
    agent = DefaultAgent(model=actor_model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    agent.step()
    second = agent.query()

    messages = second.get("extra", {}).get("verifier", {}).get("verifier_output", {}).get("input", {}).get("messages", [])
    assert [message["role"] for message in messages] == ["system", "assistant", "tool", "user"]
    assert messages[1]["content"] == "Inspect current state"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "bash"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert "<returncode>0</returncode>" in messages[2]["content"]
    assert messages[2]["tool_call_id"] == "call_1"


def test_pair_thoughts_with_toolcalls_requires_exact_num_candidates():
    config = _load_default_agent_config()
    config["candidate_sampling"] = {
        "num_candidates": 3,
        "use_n": False,
        "pair_thoughts_with_toolcalls": True,
        "sampling_kwargs": {},
    }
    config["verifier"] = {"enabled": False}

    content = "THOUGHTS: first\nTHOUGHTS: second"
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
        },
    ]
    actions = [
        {"command": "ls -la", "tool_call_id": "call_1"},
        {"command": "pwd", "tool_call_id": "call_2"},
    ]
    actor_model = DeterministicToolcallModel(outputs=[make_toolcall_output(content, tool_calls, actions)])
    agent = DefaultAgent(model=actor_model, env=LocalEnvironment(), **config)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    with pytest.raises(FormatError) as exc_info:
        agent.query()
    format_msg = (exc_info.value.messages or [{}])[0].get("content", "")
    assert "requires exactly num_candidates thought/tool-call pairs" in format_msg

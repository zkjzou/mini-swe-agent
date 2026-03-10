from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent.utils.verifier_action_evaluation import _VERIFIER_VARIANTS, evaluate_verifier_action_selection


class _VariantAwareModel:
    def __init__(self):
        self.prompts: list[str] = []

    def query(self, messages, **kwargs):
        system_prompt = messages[0].get("content", "") if messages else ""
        user_prompt = messages[-1].get("content", "") if messages else ""
        if not isinstance(system_prompt, str):
            system_prompt = ""
        if not isinstance(user_prompt, str):
            user_prompt = ""
        self.prompts.append(f"SYSTEM:\n{system_prompt}\nUSER:\n{user_prompt}")

        lower_system = system_prompt.lower()
        lower_prompt = user_prompt.lower()

        if "return yaml only" in lower_system or "top-level `rubric` list" in lower_prompt:
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: D1\n"
                    "    phase: diagnose\n"
                    "    weight: 3\n"
                    "    description: Reproduce the parser failure\n"
                    "    done_when: The failing test is reproduced locally\n"
                    "  - id: F1\n"
                    "    phase: fix\n"
                    "    weight: 2\n"
                    "    description: Implement the minimal parser fix\n"
                    "    done_when: The bug is resolved without unrelated changes\n"
                ),
                "extra": {"cost": 0.05},
            }

        if "checklist content only" in lower_system or "generate concise issue-progress checklists" in lower_system:
            return {
                "role": "assistant",
                "content": (
                    "CHECKLIST:\n"
                    "- Reproduce the parser failure\n"
                    "- Inspect the parser implementation\n"
                    "- Implement the minimal parser fix\n"
                    "- Run focused validation\n"
                ),
                "extra": {"cost": 0.05},
            }

        is_world_prompt = "predictive world model" in lower_system or "next_state:" in lower_prompt

        if "candidate action:" in lower_prompt:
            reward = "0.95" if "echo gold_target" in user_prompt else "0.10"
            if is_world_prompt:
                return {
                    "role": "assistant",
                    "content": (
                        "NEXT_STATE:\n"
                        "  Summary: Candidate advances the task.\n"
                        "  File_Changes: none\n"
                        "  Command_Output_Summary: concise output\n"
                        "  Progress: Yes + likely useful next step\n"
                        f"REWARD: {reward}"
                    ),
                    "extra": {"cost": 0.1},
                }
            if "issue progress checklist" in lower_prompt:
                return {
                    "role": "assistant",
                    "content": (
                        "CHECKLIST_ITEM_SCORES:\n"
                        "- Item 1: 1.0\n"
                        "- REASONING: candidate helps\n"
                        "- Item 2: 0.5\n"
                        "- REASONING: partial support\n"
                        "PROGRESS: Yes + advances task\n"
                        "REASONING: test\n"
                        f"SCORE: {reward}"
                    ),
                    "extra": {"cost": 0.1},
                }
            return {
                "role": "assistant",
                "content": f"REASONING: test\nFINAL: {reward}",
                "extra": {"cost": 0.1},
            }

        candidate_blocks = re.findall(r"Candidate\s+(\d+)\s*:\s*(.*?)(?=\n\s*Candidate\s+\d+\s*:|\Z)", user_prompt, re.DOTALL)
        chosen = 1
        scores: dict[int, str] = {}
        for raw_index, block in candidate_blocks:
            idx = int(raw_index)
            score = "0.95" if "echo gold_target" in block else "0.10"
            scores[idx] = score
            if score == "0.95":
                chosen = idx

        if is_world_prompt:
            content = []
            for raw_index, _block in candidate_blocks:
                idx = int(raw_index)
                content.append(
                    "\n".join(
                        [
                            f"- Candidate {idx}:",
                            "  NEXT_STATE:",
                            "    Summary: candidate effect",
                            "    File_Changes: none",
                            "    Command_Output_Summary: concise output",
                            f"    Progress: {'Yes + useful' if idx == chosen else 'No + weak step'}",
                            "  REASONING: test",
                            f"  SCORES: {scores[idx]}",
                        ]
                    )
                )
            content.append(f"REASONING: test\nFINAL: {chosen}")
            return {"role": "assistant", "content": "\n".join(content), "extra": {"cost": 0.2}}

        if "issue progress checklist" in lower_prompt:
            content = []
            for raw_index, _block in candidate_blocks:
                idx = int(raw_index)
                content.append(
                    "\n".join(
                        [
                            f"- Candidate {idx}:",
                            "  CHECKLIST_ITEM_SCORES:",
                            "  - Item 1: 1.0",
                            "  - REASONING: helpful",
                            "  - Item 2: 0.5",
                            "  - REASONING: partial",
                            f"  PROGRESS: {'Yes + useful' if idx == chosen else 'No + weak step'}",
                            "  REASONING: test",
                            f"  SCORE: {scores[idx]}",
                        ]
                    )
                )
            content.append(f"REASONING: test\nFINAL: {chosen}")
            return {"role": "assistant", "content": "\n".join(content), "extra": {"cost": 0.2}}

        content = []
        for raw_index, _block in candidate_blocks:
            idx = int(raw_index)
            content.append(f"- Candidate {idx}:\n  REASONING: test\n  SCORE: {scores[idx]}")
        content.append(f"REASONING: test\nFINAL: {chosen}")
        return {"role": "assistant", "content": "\n".join(content), "extra": {"cost": 0.2}}


def _make_action(label: str, command: str, *, is_gold: bool) -> dict:
    return {
        "label": label,
        "candidate_source": "gold" if is_gold else "sampled",
        "sampler_model_id": "gold" if is_gold else label,
        "sample_index": None if is_gold else 0,
        "is_gold": is_gold,
        "command": command,
        "model_response": {
            "role": "assistant",
            "content": f"THOUGHTS: choose this\n\nACTION:\n{command}",
            "tool_calls": [
                {
                    "id": f"tc-{label}",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }
            ],
        },
    }


def _make_row(*, run_id: str = "run-1", trajectory_relpath: str = "repo__issue-1.json") -> dict:
    return {
        "dataset_version": "verifier_candidates_v1",
        "instance_id": "repo__issue-1",
        "run_id": run_id,
        "trajectory_relpath": trajectory_relpath,
        "step_index": 3,
        "message_index": 9,
        "problem_id": "repo__issue-1",
        "history_trajectory": [
            {"role": "system", "content": "You are coding agent."},
            {"role": "user", "content": "Fix the failing parser test."},
            {"role": "assistant", "content": "SECRET_THOUGHT: inspect parser first."},
            {"role": "tool", "name": "bash", "tool_call_id": "tc-x", "content": "ok"},
        ],
        "actions": [
            _make_action("model-a", "echo bad_1", is_gold=False),
            _make_action("model-b", "echo bad_2", is_gold=False),
            _make_action("gold", "echo gold_target", is_gold=True),
            _make_action("model-c", "echo bad_3", is_gold=False),
            _make_action("model-d", "echo bad_4", is_gold=False),
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_evaluate_verifier_action_selection_defaults_to_all_variants(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    output_summary = tmp_path / "eval_summary.json"
    _write_jsonl(input_jsonl, [_make_row()])

    model = _VariantAwareModel()
    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: model,
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    assert summary["counts"]["rows_considered"] == 1
    assert summary["verifier_variants"] == [spec.name for spec in _VERIFIER_VARIANTS]
    assert set(summary["per_verifier"]) == {"first_valid", "llm", "reward_model"}
    assert summary["per_variant"]["first_valid"]["rows_evaluated"] == 1
    assert summary["per_variant"]["world_verifier"]["rows_evaluated"] == 1
    assert summary["per_variant"]["dynamic_checklist_modify_reward"]["rows_evaluated"] == 1

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == len(_VERIFIER_VARIANTS)
    assert {row["verifier_variant"] for row in rows} == {spec.name for spec in _VERIFIER_VARIANTS}
    assert all(row["status"] == "evaluated" for row in rows)
    assert all("verifier_variant" in row for row in rows)


def test_evaluate_verifier_action_selection_first_valid_does_not_require_model(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("get_model should not be called")),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=["swebench.yaml"],
        verifier_variants=["first_valid"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    assert summary["per_variant"]["first_valid"]["rows_evaluated"] == 1
    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert rows[0]["verifier_type"] == "first_valid"


def test_evaluate_verifier_action_selection_world_verifier_parses_scores(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["world_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    assert row["selected_is_gold"] is True
    assert row["verifier_output"]["scores"][row["gold_index"]] == 0.95


def test_evaluate_verifier_action_selection_basic_verifier_parses_final_and_scores(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=8,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    assert row["selected_is_gold"] is True
    assert row["verifier_output"]["raw_index"] == row["gold_index"] + 1
    assert row["verifier_output"]["scores"][row["gold_index"]] == 0.95
    assert summary["effective_max_workers"] == 1


def test_evaluate_verifier_action_selection_dynamic_checklist_metadata(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    row_1 = _make_row()
    row_2 = _make_row()
    row_2["history_trajectory"].append({"role": "tool", "name": "bash", "tool_call_id": "tc-y", "content": "updated"})
    _write_jsonl(input_jsonl, [row_1, row_2])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["dynamic_checklist_modify_reward"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 2
    first_checklist = rows[0]["verifier_output"]["checklist"]
    second_checklist = rows[1]["verifier_output"]["checklist"]
    assert first_checklist["dynamic"] is True
    assert first_checklist["update_mode"] == "modify"
    assert first_checklist["source"] == "static_checklist_seed"
    assert second_checklist["source"] == "dynamic_checklist"
    assert summary["effective_max_workers"] == 1
    assert summary["per_variant"]["dynamic_checklist_modify_reward"]["total_api_calls"] >= 4


def test_evaluate_verifier_action_selection_checklist_verifier_uses_final_selection_and_serial_execution(
    tmp_path, monkeypatch
):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["checklist_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=8,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    assert row["selected_is_gold"] is True
    assert row["verifier_output"]["raw_index"] == row["gold_index"] + 1
    assert row["verifier_output"]["scores"][row["gold_index"]] == 0.95
    assert summary["effective_max_workers"] == 1


def test_evaluate_verifier_action_selection_skips_rows_when_not_five_actions(tmp_path, monkeypatch):
    row = _make_row()
    row["actions"] = row["actions"][:4]

    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [row])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    assert summary["per_variant"]["basic_verifier"]["rows_evaluated"] == 0
    assert summary["per_variant"]["basic_verifier"]["rows_skipped"] == 1
    assert summary["per_variant"]["basic_verifier"]["skip_reasons"]["not_5_actions"] == 1

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["skip_reason"] == "not_5_actions"


def test_evaluate_verifier_action_selection_redacts_assistant_history_when_disabled(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    model = _VariantAwareModel()
    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: model,
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=tmp_path / "rows_redacted.jsonl",
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_thoughts_in_history_steps=false",
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )
    redacted_prompt_text = "\n".join(model.prompts)
    assert "SECRET_THOUGHT" not in redacted_prompt_text

    model.prompts.clear()
    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=tmp_path / "rows_unredacted.jsonl",
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_thoughts_in_history_steps=true",
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )
    unredacted_prompt_text = "\n".join(model.prompts)
    assert "SECRET_THOUGHT" in unredacted_prompt_text


def test_evaluate_verifier_action_selection_excludes_system_history_from_verifier_input(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    model = _VariantAwareModel()
    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: model,
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=tmp_path / "rows_system_filtered.jsonl",
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_variants=["basic_mini_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    prompt = "\n".join(model.prompts).lower()
    assert "fix the failing parser test." in prompt
    assert "you are coding agent." not in prompt


def test_evaluate_verifier_action_selection_can_include_verifier_and_checklist_inputs(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_inputs_in_output=true",
        ],
        verifier_variants=["checklist_reward"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    verifier_output = row["verifier_output"]
    reward_messages = verifier_output["inputs"][0]["messages"]
    assert [message["role"] for message in reward_messages] == ["system", "user"]
    assert "Task: Fix the failing parser test." in reward_messages[1]["content"]
    assert "Candidate action:" in reward_messages[1]["content"]
    checklist_messages = verifier_output["checklist"]["input"]["messages"]
    assert [message["role"] for message in checklist_messages] == ["system", "user"]
    assert "checklists" in checklist_messages[0]["content"].lower()
    assert "Issue description:" in checklist_messages[1]["content"]
    assert "Recent steps" in checklist_messages[1]["content"]


def test_evaluate_verifier_action_selection_can_include_llm_verifier_inputs(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_inputs_in_output=true",
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    messages = row["verifier_output"]["input"]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Task: Fix the failing parser test." in messages[1]["content"]


def test_evaluate_verifier_action_selection_can_use_multi_turn_verifier_history(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_inputs_in_output=true",
            'agent.verifier.history_message_format="multi_turn_chat"',
        ],
        verifier_variants=["basic_verifier"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    messages = row["verifier_output"]["input"]["messages"]
    assert [message["role"] for message in messages] == ["system", "assistant", "tool", "user"]
    assert "inspect parser first" in messages[1]["content"].lower()
    assert messages[2]["content"] == "ok"
    assert "Recent steps" not in messages[-1]["content"]
    assert "Task: Fix the failing parser test." in messages[-1]["content"]


def test_world_reward_prompt_includes_task_in_captured_input(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _VariantAwareModel(),
    )

    evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            "agent.verifier.include_inputs_in_output=true",
        ],
        verifier_variants=["world_reward"],
        strict_five_actions=True,
        show_progress=False,
        max_workers=1,
        overwrite=True,
    )

    row = json.loads(output_jsonl.read_text().splitlines()[0])
    first_candidate_messages = row["verifier_output"]["inputs"][0]["messages"]
    assert [message["role"] for message in first_candidate_messages] == ["system", "user"]
    assert any("Task: Fix the failing parser test." in message["content"] for message in first_candidate_messages)

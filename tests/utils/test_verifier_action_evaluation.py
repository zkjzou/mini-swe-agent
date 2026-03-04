from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent.utils.verifier_action_evaluation import evaluate_verifier_action_selection


class _SelectionAwareModel:
    def __init__(self):
        self.prompts: list[str] = []

    def query(self, messages, **kwargs):
        prompt = messages[-1].get("content", "") if messages else ""
        if not isinstance(prompt, str):
            prompt = ""
        self.prompts.append(prompt)

        if "Candidate action:" in prompt:
            reward = "0.95" if "echo gold_target" in prompt else "0.10"
            return {
                "role": "assistant",
                "content": f"REASONING: test\nFINAL: {reward}",
                "extra": {"cost": 0.1},
            }

        blocks = re.findall(r"Candidate\s+(\d+)\s*:\s*(.*?)(?=\n\s*Candidate\s+\d+\s*:|\Z)", prompt, re.DOTALL)
        chosen = 1
        for raw_index, block in blocks:
            if "echo gold_target" in block:
                chosen = int(raw_index)
                break
        return {
            "role": "assistant",
            "content": f"REASONING: test\nFINAL: {chosen}",
            "extra": {"cost": 0.2},
        }


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


def _make_row() -> dict:
    return {
        "dataset_version": "verifier_candidates_v1",
        "instance_id": "repo__issue-1",
        "run_id": "run-1",
        "trajectory_relpath": "repo__issue-1.json",
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


def test_evaluate_verifier_action_selection_runs_llm_and_reward_model(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    output_summary = tmp_path / "eval_summary.json"
    _write_jsonl(input_jsonl, [_make_row()])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _SelectionAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
            'agent.verifier.prompt_name="swebench/verifier"',
            'agent.verifier.prompt_dir="prompts/verifier"',
        ],
        verifier_types=["llm", "reward_model"],
        strict_five_actions=True,
        show_progress=False,
        overwrite=True,
    )

    assert summary["counts"]["rows_considered"] == 1
    assert summary["per_verifier"]["llm"]["rows_evaluated"] == 1
    assert summary["per_verifier"]["reward_model"]["rows_evaluated"] == 1
    assert summary["per_verifier"]["llm"]["gold_pick_count"] == 1
    assert summary["per_verifier"]["reward_model"]["gold_pick_count"] == 1
    assert summary["per_verifier"]["llm"]["accuracy"] == 1.0
    assert summary["per_verifier"]["reward_model"]["accuracy"] == 1.0

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["verifier_type"] for row in rows} == {"llm", "reward_model"}
    assert all(row["status"] == "evaluated" for row in rows)
    assert all(row["selected_is_gold"] is True for row in rows)


def test_evaluate_verifier_action_selection_skips_rows_when_not_five_actions(tmp_path, monkeypatch):
    row = _make_row()
    row["actions"] = row["actions"][:4]

    input_jsonl = tmp_path / "merged.jsonl"
    output_jsonl = tmp_path / "eval_rows.jsonl"
    _write_jsonl(input_jsonl, [row])

    monkeypatch.setattr(
        "minisweagent.utils.verifier_action_evaluation.get_model",
        lambda *args, **kwargs: _SelectionAwareModel(),
    )

    summary = evaluate_verifier_action_selection(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        config_specs=[
            "swebench.yaml",
            'agent.verifier.model.model_name="fake/verifier"',
            'agent.verifier.model.model_class="deterministic"',
        ],
        verifier_types=["llm"],
        strict_five_actions=True,
        show_progress=False,
        overwrite=True,
    )

    assert summary["per_verifier"]["llm"]["rows_evaluated"] == 0
    assert summary["per_verifier"]["llm"]["rows_skipped"] == 1
    assert summary["per_verifier"]["llm"]["skip_reasons"]["not_5_actions"] == 1

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["skip_reason"] == "not_5_actions"


def test_evaluate_verifier_action_selection_redacts_assistant_history_when_disabled(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "merged.jsonl"
    _write_jsonl(input_jsonl, [_make_row()])

    model = _SelectionAwareModel()
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
        verifier_types=["llm"],
        strict_five_actions=True,
        show_progress=False,
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
        verifier_types=["llm"],
        strict_five_actions=True,
        show_progress=False,
        overwrite=True,
    )
    unredacted_prompt_text = "\n".join(model.prompts)
    assert "SECRET_THOUGHT" in unredacted_prompt_text

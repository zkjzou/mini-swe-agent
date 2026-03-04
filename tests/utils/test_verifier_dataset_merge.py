from __future__ import annotations

import json
from pathlib import Path

import pytest

from minisweagent.utils.verifier_dataset_merge import merge_verifier_sampling_datasets


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _base_row(**kwargs):
    row = {
        "instance_id": "repo__issue-1",
        "run_id": "run-1",
        "trajectory_relpath": "repo__issue-1.json",
        "step_index": 0,
        "message_index": 12,
        "problem_id": "repo__issue-1",
        "sampler_model_id": "sampler-a",
        "sample_index": 0,
        "candidate_source": "sampled",
        "is_gold": False,
        "prompt_messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "task prompt"},
        ],
        "candidate_message": {"role": "assistant", "content": "sampled response", "tool_calls": []},
        "actions": [{"command": "echo hi"}],
    }
    row.update(kwargs)
    return row


def _find_action(actions: list[dict], label: str) -> dict | None:
    for action in actions:
        if action.get("label") == label:
            return action
    return None


def test_grouped_merge_builds_one_row_per_run_step_and_labeled_actions(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    merged = tmp_path / "merged.jsonl"

    gold = _base_row(
        candidate_source="gold",
        sampler_model_id="gold",
        sample_index=None,
        is_gold=True,
        candidate_message={"role": "assistant", "content": "gold response", "tool_calls": []},
        actions=[{"command": "ls"}],
    )
    row_a = _base_row(sampler_model_id="sampler-a", actions=[{"command": "echo A"}])
    row_b = _base_row(sampler_model_id="sampler-b", actions=[{"command": "echo B"}])

    _write_jsonl(file_a, [gold, row_a])
    _write_jsonl(file_b, [row_b])

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a, file_b],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_first",
        require_gold=True,
    )

    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]

    assert row["run_id"] == "run-1"
    assert row["step_index"] == 0
    assert row["n_actions"] == 3
    assert len(row["history_trajectory"]) == 2
    assert row["history_trajectory"][0]["role"] == "system"
    labels = [action["label"] for action in row["actions"]]
    assert labels[0] == "gold"
    assert set(labels) == {"gold", "sampler-a", "sampler-b"}
    assert row["actions"][0]["command"] == "ls"
    assert row["actions"][0]["model_response"]["content"] == "gold response"
    sampled_action = _find_action(row["actions"], "sampler-a")
    assert sampled_action is not None
    assert sampled_action["model_response"]["content"] == "sampled response"
    assert "gold" in row["candidates_by_source"]
    assert "sampler-a" in row["candidates_by_source"]
    assert "sampler-b" in row["candidates_by_source"]
    # per-source payload should stay compact
    assert "prompt_messages" not in row["candidates_by_source"]["gold"]
    assert "candidate_message" not in row["candidates_by_source"]["gold"]
    assert row["candidates_by_source"]["gold"]["has_actions"] is True

    assert summary["counts"]["rows_kept"] == 1
    assert summary["counts"]["groups_kept"] == 1
    assert summary["counts"]["actions_kept_total"] == 3


def test_grouped_merge_key_includes_run_identity(tmp_path):
    file_a = tmp_path / "a.jsonl"
    merged = tmp_path / "merged.jsonl"

    rows = [
        _base_row(
            run_id="run-1",
            trajectory_relpath="run1.json",
            candidate_source="gold",
            sampler_model_id="gold",
            sample_index=None,
            is_gold=True,
            actions=[{"command": "gold-1"}],
        ),
        _base_row(run_id="run-1", trajectory_relpath="run1.json", sampler_model_id="sampler-a", actions=[{"command": "a-1"}]),
        _base_row(
            run_id="run-2",
            trajectory_relpath="run2.json",
            candidate_source="gold",
            sampler_model_id="gold",
            sample_index=None,
            is_gold=True,
            actions=[{"command": "gold-2"}],
        ),
        _base_row(run_id="run-2", trajectory_relpath="run2.json", sampler_model_id="sampler-a", actions=[{"command": "a-2"}]),
    ]

    _write_jsonl(file_a, rows)

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_first",
        require_gold=True,
    )

    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(merged_rows) == 2
    assert summary["counts"]["conflicts"] == 0
    run_ids = {row["run_id"] for row in merged_rows}
    assert run_ids == {"run-1", "run-2"}


def test_grouped_merge_keep_last_for_conflicts(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    merged = tmp_path / "merged.jsonl"

    gold = _base_row(
        candidate_source="gold",
        sampler_model_id="gold",
        sample_index=None,
        is_gold=True,
        actions=[{"command": "ls"}],
    )
    row_first = _base_row(sampler_model_id="sampler-a", actions=[{"command": "echo first"}])
    row_last = _base_row(sampler_model_id="sampler-a", actions=[{"command": "echo last"}])

    _write_jsonl(file_a, [gold, row_first])
    _write_jsonl(file_b, [row_last])

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a, file_b],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_last",
        require_gold=True,
    )

    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(rows) == 1
    action = _find_action(rows[0]["actions"], "sampler-a")
    assert action is not None
    assert action["command"] == "echo last"
    assert summary["counts"]["conflicts"] == 1


def test_grouped_merge_error_on_conflict(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    merged = tmp_path / "merged.jsonl"

    row_a = _base_row(actions=[{"command": "echo first"}])
    row_b = _base_row(actions=[{"command": "echo second"}])
    _write_jsonl(file_a, [row_a])
    _write_jsonl(file_b, [row_b])

    with pytest.raises(ValueError, match="Conflict for key"):
        merge_verifier_sampling_datasets(
            input_paths=[file_a, file_b],
            output_jsonl=merged,
            dedupe="semantic_key",
            conflict_policy="error",
            require_gold=False,
        )


def test_grouped_merge_require_gold_drops_group_without_gold(tmp_path):
    file_a = tmp_path / "a.jsonl"
    merged = tmp_path / "merged.jsonl"

    sampled_only = _base_row(sampler_model_id="sampler-a", actions=[{"command": "echo sampled"}])
    _write_jsonl(file_a, [sampled_only])

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_first",
        require_gold=True,
    )

    assert merged.read_text().strip() == ""
    assert summary["counts"]["missing_gold_steps"] == 1
    assert summary["counts"]["rows_kept"] == 0


def test_grouped_merge_uses_sample_index_zero_only(tmp_path):
    file_a = tmp_path / "a.jsonl"
    merged = tmp_path / "merged.jsonl"

    rows = [
        _base_row(
            candidate_source="gold",
            sampler_model_id="gold",
            sample_index=None,
            is_gold=True,
            actions=[{"command": "ls"}],
        ),
        _base_row(sampler_model_id="sampler-a", sample_index=0, actions=[{"command": "echo s0"}]),
        _base_row(sampler_model_id="sampler-a", sample_index=1, actions=[{"command": "echo s1"}]),
    ]
    _write_jsonl(file_a, rows)

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a],
        output_jsonl=merged,
        dedupe="none",
        conflict_policy="keep_first",
        require_gold=True,
    )

    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(merged_rows) == 1
    action = _find_action(merged_rows[0]["actions"], "sampler-a")
    assert action is not None
    assert action["command"] == "echo s0"
    assert summary["counts"]["nonzero_sample_rows_ignored"] == 1


def test_grouped_merge_drops_missing_sampled_actions_from_actions_list(tmp_path):
    file_a = tmp_path / "a.jsonl"
    merged = tmp_path / "merged.jsonl"

    rows = [
        _base_row(
            candidate_source="gold",
            sampler_model_id="gold",
            sample_index=None,
            is_gold=True,
            actions=[{"command": "ls"}],
        ),
        _base_row(sampler_model_id="sampler-a", actions=[]),
        _base_row(sampler_model_id="sampler-b", actions=[{"command": "echo b"}]),
    ]
    _write_jsonl(file_a, rows)

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_first",
        require_gold=True,
    )

    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(merged_rows) == 1
    row = merged_rows[0]
    labels = [entry["label"] for entry in row["actions"]]
    assert labels == ["gold", "sampler-b"]
    assert "sampler-a" in row["missing_model_sources"]
    assert row["n_actions"] == 2
    assert summary["counts"]["actions_kept_sampled"] == 1

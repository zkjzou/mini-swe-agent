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
        "message_index": 12,
        "sampler_model_id": "sampler-a",
        "sample_index": 0,
        "candidate_source": "sampled",
        "is_gold": False,
        "actions": [{"command": "echo hi"}],
    }
    row.update(kwargs)
    return row


def test_merge_semantic_key_keep_first_and_require_gold(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    merged = tmp_path / "merged.jsonl"

    gold = _base_row(candidate_source="gold", sampler_model_id="gold", sample_index=None, is_gold=True, actions=[{"command": "ls"}])
    row_a = _base_row(actions=[{"command": "echo A"}])
    conflict_row = _base_row(actions=[{"command": "echo B"}])
    row_b = _base_row(sampler_model_id="sampler-b", sample_index=0, actions=[{"command": "echo C"}])
    no_gold_step = _base_row(instance_id="repo__issue-2", message_index=4, sampler_model_id="sampler-a")

    _write_jsonl(file_a, [gold, row_a])
    _write_jsonl(file_b, [conflict_row, row_b, no_gold_step])

    summary = merge_verifier_sampling_datasets(
        input_paths=[file_a, file_b],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_first",
        require_gold=True,
    )

    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(rows) == 3
    assert any(row["candidate_source"] == "gold" for row in rows)
    sampler_a_rows = [row for row in rows if row["sampler_model_id"] == "sampler-a" and row["candidate_source"] == "sampled"]
    assert sampler_a_rows[0]["actions"][0]["command"] == "echo A"
    assert summary["counts"]["conflicts"] == 1
    assert summary["counts"]["missing_gold_steps"] == 1


def test_merge_semantic_key_keep_last(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    merged = tmp_path / "merged.jsonl"

    gold = _base_row(candidate_source="gold", sampler_model_id="gold", sample_index=None, is_gold=True, actions=[{"command": "ls"}])
    row_a = _base_row(actions=[{"command": "echo first"}])
    row_b = _base_row(actions=[{"command": "echo last"}])

    _write_jsonl(file_a, [gold, row_a])
    _write_jsonl(file_b, [row_b])

    merge_verifier_sampling_datasets(
        input_paths=[file_a, file_b],
        output_jsonl=merged,
        dedupe="semantic_key",
        conflict_policy="keep_last",
        require_gold=True,
    )

    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    sampled = [row for row in rows if row["candidate_source"] == "sampled"]
    assert sampled[0]["actions"][0]["command"] == "echo last"


def test_merge_semantic_key_error_on_conflict(tmp_path):
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

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.utils.verifier_action_sampling import (
    extract_replay_steps,
    generate_verifier_sampling_dataset,
    normalize_docent_message_for_model,
    trajectory_contains_parallel_tool_calls,
)


class _FakeSamplerModel:
    def query(self, messages, **kwargs):
        return {
            "role": "assistant",
            "content": "sampled candidate",
            "extra": {
                "actions": [{"command": "echo sampled"}],
                "cost": 0.01,
                "response": {"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}},
            },
        }


class _FailingSamplerModel:
    def query(self, messages, **kwargs):
        raise RuntimeError("boom")


def _write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "problem_id": "repo__issue-1",
                "transcript": {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant",
                            "content": "first action",
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "function": "bash",
                                    "arguments": {"command": "ls -la"},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "tc1",
                            "function": "bash",
                            "content": "ok",
                        },
                        {
                            "role": "assistant",
                            "content": "second action",
                            "tool_calls": [
                                {
                                    "id": "tc2",
                                    "function": "bash",
                                    "arguments": {"command": "pwd"},
                                }
                            ],
                        },
                    ]
                },
            }
        )
    )


def test_normalize_docent_message_for_model_preserves_tool_calls():
    message = {
        "role": "assistant",
        "content": "do action",
        "tool_calls": [{"id": "tc1", "function": "bash", "arguments": {"command": "ls -la"}}],
    }
    normalized = normalize_docent_message_for_model(message)
    assert normalized["role"] == "assistant"
    assert normalized["tool_calls"][0]["function"]["name"] == "bash"
    assert normalized["tool_calls"][0]["function"]["arguments"] == '{"command": "ls -la"}'


def test_extract_replay_steps_only_assistant_toolcall_steps():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "no action"},
        {"role": "assistant", "content": "action", "tool_calls": [{"id": "1", "function": "bash", "arguments": {"command": "ls"}}]},
    ]
    steps = extract_replay_steps(messages)
    assert len(steps) == 1
    assert steps[0]["message_index"] == 2


def test_trajectory_contains_parallel_tool_calls():
    messages = [
        {"role": "assistant", "content": "single", "tool_calls": [{"id": "1", "function": "bash", "arguments": {"command": "ls"}}]},
        {
            "role": "assistant",
            "content": "parallel",
            "tool_calls": [
                {"id": "2", "function": "bash", "arguments": {"command": "pwd"}},
                {"id": "3", "function": "bash", "arguments": {"command": "whoami"}},
            ],
        },
    ]
    assert trajectory_contains_parallel_tool_calls(messages) is True


def test_generate_verifier_sampling_dataset(tmp_path, monkeypatch):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript_path = transcripts_dir / "repo__issue-1.json"
    _write_transcript(transcript_path)

    output_json = tmp_path / "output.json"
    output_json.write_text(
        json.dumps(
            [
                {
                    "id": "run-id",
                    "metadata": {
                        "instance_id": "repo__issue-1",
                        "scores": {"resolved": 1},
                    },
                    "transcripts": [str(transcript_path.name)],
                },
                {
                    "id": "run-unresolved",
                    "metadata": {
                        "instance_id": "repo__issue-2",
                        "scores": {"resolved": 0},
                    },
                    "transcripts": ["repo__issue-2.json"],
                },
            ]
        )
    )

    sampler_config = tmp_path / "samplers.yaml"
    sampler_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "sampler-a",
                        "model_name": "fake/model",
                        "sampling_kwargs": {"temperature": 0.8},
                    }
                ]
            }
        )
    )

    monkeypatch.setattr("minisweagent.utils.verifier_action_sampling.get_model", lambda *args, **kwargs: _FakeSamplerModel())
    out_dir = tmp_path / "dataset"
    summary = generate_verifier_sampling_dataset(
        output_json_path=output_json,
        transcripts_dir=transcripts_dir,
        sampler_config_path=sampler_config,
        output_dir=out_dir,
        num_samples=2,
        max_workers=2,
    )

    assert summary["counts"]["runs_processed"] == 1
    assert summary["counts"]["planned_runs"] == 1
    assert summary["counts"]["steps_processed"] == 2
    assert summary["counts"]["planned_steps"] == 2
    assert summary["counts"]["planned_sample_calls"] == 4
    assert summary["counts"]["gold_candidates"] == 2
    assert summary["counts"]["sample_candidates"] == 4

    rows = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert any(row["is_gold"] for row in rows)
    sampled_rows = [row for row in rows if row["candidate_source"] == "sampled"]
    assert all(row["actions"][0]["command"] == "echo sampled" for row in sampled_rows)


def test_generate_verifier_sampling_dataset_records_sampling_errors(tmp_path, monkeypatch):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript_path = transcripts_dir / "repo__issue-1.json"
    _write_transcript(transcript_path)

    output_json = tmp_path / "output.json"
    output_json.write_text(
        json.dumps(
            [
                {
                    "id": "run-id",
                    "metadata": {
                        "instance_id": "repo__issue-1",
                        "scores": {"resolved": 1},
                    },
                    "transcripts": [str(transcript_path.name)],
                }
            ]
        )
    )

    sampler_config = tmp_path / "samplers.yaml"
    sampler_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "sampler-a",
                        "model_name": "fake/model",
                    }
                ]
            }
        )
    )

    monkeypatch.setattr("minisweagent.utils.verifier_action_sampling.get_model", lambda *args, **kwargs: _FailingSamplerModel())
    out_dir = tmp_path / "dataset"
    summary = generate_verifier_sampling_dataset(
        output_json_path=output_json,
        transcripts_dir=transcripts_dir,
        sampler_config_path=sampler_config,
        output_dir=out_dir,
        num_samples=1,
        max_workers=1,
    )
    assert summary["counts"]["sample_failures"] == 2
    rows = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text().splitlines()]
    sampled_rows = [row for row in rows if row["candidate_source"] == "sampled"]
    assert all(row["error"] is not None for row in sampled_rows)


def test_generate_verifier_sampling_dataset_skips_parallel_tool_call_trajectories(tmp_path, monkeypatch):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript_path = transcripts_dir / "repo__issue-1.json"
    transcript_path.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "problem_id": "repo__issue-1",
                "transcript": {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant",
                            "content": "parallel action",
                            "tool_calls": [
                                {"id": "tc1", "function": "bash", "arguments": {"command": "ls -la"}},
                                {"id": "tc2", "function": "bash", "arguments": {"command": "pwd"}},
                            ],
                        },
                    ]
                },
            }
        )
    )

    output_json = tmp_path / "output.json"
    output_json.write_text(
        json.dumps(
            [
                {
                    "id": "run-id",
                    "metadata": {"instance_id": "repo__issue-1", "scores": {"resolved": 1}},
                    "transcripts": [str(transcript_path.name)],
                }
            ]
        )
    )

    sampler_config = tmp_path / "samplers.yaml"
    sampler_config.write_text(json.dumps({"models": [{"id": "sampler-a", "model_name": "fake/model"}]}))
    monkeypatch.setattr("minisweagent.utils.verifier_action_sampling.get_model", lambda *args, **kwargs: _FakeSamplerModel())

    out_dir = tmp_path / "dataset"
    summary = generate_verifier_sampling_dataset(
        output_json_path=output_json,
        transcripts_dir=transcripts_dir,
        sampler_config_path=sampler_config,
        output_dir=out_dir,
        num_samples=1,
        max_workers=1,
    )

    assert summary["counts"]["runs_processed"] == 0
    assert summary["counts"]["runs_skipped_parallel_tool_calls"] == 1
    assert (out_dir / "candidates.jsonl").read_text().strip() == ""

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


class _CountingSamplerModel:
    def __init__(self):
        self.calls = 0

    def query(self, messages, **kwargs):
        self.calls += 1
        return {
            "role": "assistant",
            "content": f"sampled candidate {self.calls}",
            "extra": {
                "actions": [{"command": f"echo resampled-{self.calls}"}],
                "cost": 0.01,
                "response": {"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}},
            },
        }


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




def test_normalize_docent_message_for_model_drops_reasoning_items():
    message = {
        "role": "assistant",
        "content": [
            {"type": "reasoning_text", "text": "think privately"},
            {"type": "output_text", "text": "visible text"},
        ],
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "hidden"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "reasoning_text", "text": "hidden"},
                    {"type": "output_text", "text": "keep me"},
                ],
            },
        ],
    }

    normalized = normalize_docent_message_for_model(message)

    assert normalized["content"] == [{"type": "output_text", "text": "visible text"}]
    assert normalized["output"] == [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "keep me"}]}
    ]


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


def test_generate_verifier_sampling_dataset_resamples_only_invalid_existing_slots(tmp_path, monkeypatch):
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
    sampler_config.write_text(json.dumps({"models": [{"id": "sampler-a", "model_name": "fake/model"}]}))

    out_dir = tmp_path / "dataset"
    out_dir.mkdir()
    existing_rows = [
        {
            "dataset_version": "verifier_candidates_v1",
            "instance_id": "repo__issue-1",
            "run_id": "run-id",
            "problem_id": "repo__issue-1",
            "trajectory_relpath": transcript_path.name,
            "step_index": 0,
            "message_index": 2,
            "candidate_source": "sampled",
            "is_gold": False,
            "sampler_model_id": "sampler-a",
            "sampler_model_name": "fake/model",
            "sample_index": 0,
            "prompt_messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
            "candidate_message": {"role": "assistant", "content": "kept"},
            "actions": [{"command": "echo kept"}],
            "action_text": "echo kept",
            "has_actions": True,
            "error": None,
            "usage": None,
            "created_at": 1,
        },
        {
            "dataset_version": "verifier_candidates_v1",
            "instance_id": "repo__issue-1",
            "run_id": "run-id",
            "problem_id": "repo__issue-1",
            "trajectory_relpath": transcript_path.name,
            "step_index": 1,
            "message_index": 4,
            "candidate_source": "sampled",
            "is_gold": False,
            "sampler_model_id": "sampler-a",
            "sampler_model_name": "fake/model",
            "sample_index": 0,
            "prompt_messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
            "candidate_message": {"role": "assistant", "content": "invalid"},
            "actions": [],
            "action_text": "",
            "has_actions": False,
            "error": None,
            "usage": None,
            "created_at": 2,
        },
    ]
    (out_dir / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in existing_rows) + "\n")
    (out_dir / "summary.json").write_text("{}")

    model = _CountingSamplerModel()
    monkeypatch.setattr("minisweagent.utils.verifier_action_sampling.get_model", lambda *args, **kwargs: model)

    summary = generate_verifier_sampling_dataset(
        output_json_path=output_json,
        transcripts_dir=transcripts_dir,
        sampler_config_path=sampler_config,
        output_dir=out_dir,
        num_samples=1,
        max_workers=1,
        show_progress=False,
        resample_invalid_only=True,
    )

    assert model.calls == 1
    assert summary["counts"]["planned_sample_slots"] == 2
    assert summary["counts"]["planned_sample_calls"] == 1
    assert summary["counts"]["sample_candidates"] == 2
    assert summary["counts"]["sample_candidates_preserved"] == 1
    assert summary["counts"]["sample_candidates_resampled"] == 1

    rows = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text().splitlines()]
    sampled_rows = [row for row in rows if row["candidate_source"] == "sampled"]
    assert len(sampled_rows) == 2
    sampled_by_step = {row["step_index"]: row for row in sampled_rows}
    assert sampled_by_step[0]["actions"][0]["command"] == "echo kept"
    assert sampled_by_step[1]["actions"][0]["command"] == "echo resampled-1"

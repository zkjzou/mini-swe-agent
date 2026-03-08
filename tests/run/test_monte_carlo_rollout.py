from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.extra.monte_carlo import app, generate_monte_carlo_rollouts
from minisweagent.run.extra.utils.trajectory_replay import build_candidate_branch_message, seed_agent_from_history
from minisweagent.run.utilities.mini_extra import main as mini_extra_main


def _write_config(path: Path, cwd: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "system_template": "sys",
                    "instance_template": "{{ task }}",
                    "cost_limit": 0,
                    "step_limit": 0,
                    "candidate_sampling": {"num_candidates": 1, "use_n": False, "sampling_kwargs": {}},
                    "verifier": {"enabled": False},
                },
                "environment": {
                    "environment_class": "local",
                    "cwd": str(cwd),
                    "timeout": 5,
                },
                "model": {
                    "model_name": "deterministic_toolcall",
                    "model_class": "deterministic_toolcall",
                },
            }
        )
    )


def _make_row() -> dict:
    return {
        "instance_id": "repo__issue-1",
        "problem_id": "repo__issue-1",
        "run_id": "run-1",
        "trajectory_relpath": "repo__issue-1.json",
        "step_index": 1,
        "message_index": 3,
        "history_trajectory": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Fix the issue"},
            {
                "role": "assistant",
                "content": "Inspect workspace first.",
                "tool_calls": [
                    {
                        "id": "tc_hist",
                        "function": "bash",
                        "arguments": {"command": "printf replayed > replay.txt"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc_hist",
                "name": "bash",
                "content": "stale recorded output",
            },
        ],
        "actions": [
            {
                "label": "gold",
                "command": "printf gold > branch.txt",
                "is_gold": True,
                "tool_call_id": "tc_gold",
                "candidate_source": "gold",
                "model_response": {"role": "assistant", "content": "Gold thought"},
            },
            {
                "label": "alt",
                "command": "printf alt > branch.txt",
                "is_gold": False,
                "tool_call_id": "tc_alt",
                "candidate_source": "sampled",
                "model_response": {"role": "assistant", "content": "Alt thought"},
            },
        ],
    }


def _make_submit_model() -> DeterministicToolcallModel:
    return DeterministicToolcallModel(
        outputs=[
            make_toolcall_output(
                "Submit the patch.",
                tool_calls=[
                    {
                        "id": "tc_submit",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "printf \'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\\\npatch\'"}',
                        },
                    }
                ],
                actions=[
                    {
                        "command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch'",
                        "tool_call_id": "tc_submit",
                    }
                ],
            )
        ]
    )


def test_seed_agent_from_history_replays_live_commands(tmp_path):
    from minisweagent.agents import get_agent
    from minisweagent.models.test_models import DeterministicToolcallModel

    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    agent = get_agent(
        DeterministicToolcallModel(outputs=[]),
        __import__("minisweagent.environments", fromlist=["get_environment"]).get_environment(config["environment"]),
        config["agent"],
        default_type="default",
    )

    summary = seed_agent_from_history(agent, _make_row())

    assert summary.replayed_prefix_steps == 1
    assert (tmp_path / "replay.txt").read_text() == "replayed"
    assert "stale recorded output" not in json.dumps(agent.messages)


def test_build_candidate_branch_message_includes_forced_action():
    message = build_candidate_branch_message(_make_row()["actions"][0], action_index=0)
    assert message["role"] == "assistant"
    assert message["extra"]["actions"][0]["command"] == "printf gold > branch.txt"
    assert message["extra"]["candidate"]["label"] == "gold"


def test_generate_monte_carlo_rollouts(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")

    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.load_dataset", lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}])
    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.get_model", lambda *args, **kwargs: _make_submit_model())

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=2,
        max_rollout_steps=1,
        max_workers=1,
        show_progress=False,
    )

    assert summary["counts"]["rows_loaded"] == 1
    assert summary["counts"]["tasks_planned"] == 4
    assert summary["counts"]["tasks_completed"] == 4
    assert summary["counts"]["solved"] == 4

    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {row["action_label"] for row in rows} == {"gold", "alt"}
    assert all(row["rollout_exit_status"] == "Submitted" for row in rows)
    assert all(Path(row["trajectory_path"]).exists() for row in rows)


def test_monte_carlo_cli_invokes_generator(monkeypatch, tmp_path):
    called = {}

    def _fake_generate(**kwargs):
        called.update(kwargs)
        return {
            "results_jsonl": str(tmp_path / "out" / "results.jsonl"),
            "counts": {
                "rows_loaded": 1,
                "tasks_planned": 2,
                "tasks_completed": 2,
                "solved": 1,
                "replay_failures": 0,
            },
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.generate_monte_carlo_rollouts", _fake_generate)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            str(tmp_path / "merged.jsonl"),
            "--subset",
            "verified",
            "--split",
            "dev",
            "--output-dir",
            str(tmp_path / "out"),
            "--samples-per-action",
            "3",
            "--max-rollout-steps",
            "7",
            "--max-workers",
            "5",
            "--limit-rows",
            "2",
            "--instance",
            "repo__issue-1",
            "--step-index",
            "4",
            "--no-show-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["samples_per_action"] == 3
    assert called["max_rollout_steps"] == 7
    assert called["max_workers"] == 5
    assert called["limit_rows"] == 2
    assert called["instance_filter"] == ["repo__issue-1"]
    assert called["step_index"] == 4


def test_mini_extra_dispatch_exposes_monte_carlo(monkeypatch):
    called = {}

    def _fake_app(args=None, prog_name=None):
        called["args"] = args
        called["prog_name"] = prog_name

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.app", _fake_app)
    monkeypatch.setattr("sys.argv", ["mini-extra", "monte-carlo-rollout", "--help"])

    mini_extra_main()

    assert called["args"] == ["--help"]
    assert called["prog_name"] == "mini-extra monte-carlo-rollout"

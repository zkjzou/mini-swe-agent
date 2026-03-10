from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.extra.monte_carlo import _load_rows, app, generate_monte_carlo_rollouts
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


def _write_existing_results(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _write_placeholder_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _write_existing_trajectory(
    path: Path,
    *,
    instance_id: str = "repo__issue-1",
    step_index: int = 1,
    action_index: int = 0,
    action_label: str = "gold",
    sample_index: int = 0,
    submission: str = "old_patch",
    replay_status: str = "ok",
    rollout_exit_status: str = "Submitted",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trajectory_format": "messages",
                "instance_id": instance_id,
                "messages": [],
                "info": {
                    "exit_status": rollout_exit_status,
                    "submission": submission,
                    "rollout": {
                        "source_jsonl_line": 1,
                        "instance_id": instance_id,
                        "source_step_index": step_index,
                        "source_message_index": 3,
                        "trajectory_relpath": f"{instance_id}.json",
                        "action_index": action_index,
                        "action_label": action_label,
                        "candidate_source": "gold" if action_label == "gold" else "sampled",
                        "is_gold": action_label == "gold",
                        "sample_index": sample_index,
                        "forced_command": "printf old > branch.txt",
                        "forced_thought": "existing thought",
                        "replay_status": replay_status,
                        "replayed_prefix_steps": 1,
                        "rollout_exit_status": rollout_exit_status,
                        "rollout_executed_steps": 0,
                        "cost": 0.0,
                        "agent_api_calls": 0,
                        "verifier_api_calls": 0,
                        "duration_seconds": 0.1,
                    },
                },
            }
        )
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

    preds = json.loads((tmp_path / "out" / "preds.json").read_text())
    assert preds == {
        "repo__issue-1__step_0001__action_00__gold__sample_000": {
            "model_name_or_path": "deterministic_toolcall",
            "instance_id": "repo__issue-1__step_0001__action_00__gold__sample_000",
            "model_patch": "patch",
        },
        "repo__issue-1__step_0001__action_00__gold__sample_001": {
            "model_name_or_path": "deterministic_toolcall",
            "instance_id": "repo__issue-1__step_0001__action_00__gold__sample_001",
            "model_patch": "patch",
        },
        "repo__issue-1__step_0001__action_01__alt__sample_000": {
            "model_name_or_path": "deterministic_toolcall",
            "instance_id": "repo__issue-1__step_0001__action_01__alt__sample_000",
            "model_patch": "patch",
        },
        "repo__issue-1__step_0001__action_01__alt__sample_001": {
            "model_name_or_path": "deterministic_toolcall",
            "instance_id": "repo__issue-1__step_0001__action_01__alt__sample_001",
            "model_patch": "patch",
        },
    }
    preds_by_instance = json.loads((tmp_path / "out" / "preds_by_instance.json").read_text())
    assert preds_by_instance == {
        "repo__issue-1": {
            "model_name_or_path": "deterministic_toolcall",
            "instance_id": "repo__issue-1",
            "model_patch": "patch",
        }
    }


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
            "--row-start",
            "3",
            "--row-end",
            "7",
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
    assert called["row_start"] == 3
    assert called["row_end"] == 7
    assert called["instance_filter"] == ["repo__issue-1"]
    assert called["step_index"] == 4


def test_load_rows_applies_row_start_and_row_end_after_filters(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = []
    for index in range(1, 7):
        row = _make_row()
        row["instance_id"] = "keep" if index != 3 else "skip"
        row["step_index"] = 1 if index != 4 else 2
        row["message_index"] = index
        row["run_id"] = f"run-{index}"
        rows.append(row)
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = _load_rows(
        rows_path,
        instance_filter={"keep"},
        step_index=1,
        row_start=2,
        row_end=3,
        limit_rows=None,
    )

    assert [line_no for line_no, _ in loaded] == [2, 5]
    assert [row["run_id"] for _, row in loaded] == ["run-2", "run-5"]


def test_generate_monte_carlo_rollouts_skips_existing_by_default(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    existing_record = {
        "task_key": "repo__issue-1::1::0::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 0,
        "sample_index": 0,
        "action_label": "gold",
        "is_gold": True,
        "submission": "old_patch",
        "rollout_exit_status": "Submitted",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "existing.traj.json"),
        "error": None,
    }
    _write_placeholder_file(Path(existing_record["trajectory_path"]))
    _write_existing_results(tmp_path / "out" / "results.jsonl", [existing_record])

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.load_dataset", lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}])

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": f"repo__issue-1::1::{kwargs['action_index']}::{kwargs['sample_index']}",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": kwargs["action_index"],
            "sample_index": kwargs["sample_index"],
            "action_label": "alt" if kwargs["action_index"] else "gold",
            "is_gold": kwargs["action_index"] == 0,
            "submission": "new_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / f"{kwargs['action_index']}_{kwargs['sample_index']}.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        show_progress=False,
    )

    assert called["count"] == 1
    assert summary["counts"]["tasks_skipped_existing"] == 1
    preds = json.loads((tmp_path / "out" / "preds.json").read_text())
    assert preds["repo__issue-1__step_0001__action_00__gold__sample_000"]["model_patch"] == "old_patch"


def test_generate_monte_carlo_rollouts_redo_errors_only(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    ok_record = {
        "task_key": "repo__issue-1::1::0::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 0,
        "sample_index": 0,
        "action_label": "gold",
        "is_gold": True,
        "submission": "old_patch",
        "rollout_exit_status": "Submitted",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "ok.traj.json"),
        "error": None,
    }
    err_record = {
        "task_key": "repo__issue-1::1::1::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 1,
        "sample_index": 0,
        "action_label": "alt",
        "is_gold": False,
        "submission": "",
        "rollout_exit_status": "RuntimeError",
        "replay_status": "error",
        "trajectory_path": str(tmp_path / "out" / "missing.traj.json"),
        "error": {"type": "RuntimeError", "message": "boom"},
    }
    _write_placeholder_file(Path(ok_record["trajectory_path"]))
    _write_existing_results(tmp_path / "out" / "results.jsonl", [ok_record, err_record])

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.load_dataset", lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}])

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": "repo__issue-1::1::1::0",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": 1,
            "sample_index": 0,
            "action_label": "alt",
            "is_gold": False,
            "submission": "new_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / "rerun.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        redo_errors=True,
        show_progress=False,
    )

    assert called["count"] == 1
    assert summary["counts"]["tasks_skipped_existing"] == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    rerun = next(row for row in rows if row["action_index"] == 1)
    assert rerun["submission"] == "new_patch"


def test_generate_monte_carlo_rollouts_redo_exit_status_only(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    status_record = {
        "task_key": "repo__issue-1::1::0::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 0,
        "sample_index": 0,
        "action_label": "gold",
        "is_gold": True,
        "submission": "",
        "rollout_exit_status": "InternalServerError",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "status.traj.json"),
        "error": None,
    }
    ok_record = {
        "task_key": "repo__issue-1::1::1::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 1,
        "sample_index": 0,
        "action_label": "alt",
        "is_gold": False,
        "submission": "old_patch",
        "rollout_exit_status": "Submitted",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "ok.traj.json"),
        "error": None,
    }
    _write_placeholder_file(Path(status_record["trajectory_path"]))
    _write_placeholder_file(Path(ok_record["trajectory_path"]))
    _write_existing_results(tmp_path / "out" / "results.jsonl", [status_record, ok_record])

    monkeypatch.setattr(
        "minisweagent.run.extra.monte_carlo.load_dataset",
        lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}],
    )

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": "repo__issue-1::1::0::0",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": 0,
            "sample_index": 0,
            "action_label": "gold",
            "is_gold": True,
            "submission": "rerun_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / "rerun.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        redo_exit_status=["InternalServerError", "BadRequestError"],
        show_progress=False,
    )

    assert called["count"] == 1
    assert summary["counts"]["tasks_skipped_existing"] == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    rerun = next(row for row in rows if row["action_index"] == 0)
    assert rerun["submission"] == "rerun_patch"


def test_generate_monte_carlo_rollouts_redo_existing_reruns_all(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    old_record = {
        "task_key": "repo__issue-1::1::0::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 0,
        "sample_index": 0,
        "action_label": "gold",
        "is_gold": True,
        "submission": "old_patch",
        "rollout_exit_status": "Submitted",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "old.traj.json"),
        "error": None,
    }
    _write_placeholder_file(Path(old_record["trajectory_path"]))
    _write_existing_results(tmp_path / "out" / "results.jsonl", [old_record])

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.load_dataset", lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}])

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": f"repo__issue-1::1::{kwargs['action_index']}::{kwargs['sample_index']}",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": kwargs["action_index"],
            "sample_index": kwargs["sample_index"],
            "action_label": "alt" if kwargs["action_index"] else "gold",
            "is_gold": kwargs["action_index"] == 0,
            "submission": "rerun_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / f"rerun_{kwargs['action_index']}.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        redo_existing=True,
        show_progress=False,
    )

    assert called["count"] == 2
    assert summary["counts"]["tasks_skipped_existing"] == 0


def test_generate_monte_carlo_rollouts_backfills_existing_from_trajectories(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    existing_traj = tmp_path / "out" / "repo__issue-1" / "step_0001" / "gold__sample_000.traj.json"
    _write_existing_trajectory(existing_traj)

    monkeypatch.setattr(
        "minisweagent.run.extra.monte_carlo.load_dataset",
        lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}],
    )

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": f"repo__issue-1::1::{kwargs['action_index']}::{kwargs['sample_index']}",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": kwargs["action_index"],
            "sample_index": kwargs["sample_index"],
            "action_label": "alt" if kwargs["action_index"] else "gold",
            "is_gold": kwargs["action_index"] == 0,
            "submission": "new_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / f"{kwargs['action_index']}_{kwargs['sample_index']}.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        show_progress=False,
    )

    assert called["count"] == 1
    assert summary["counts"]["existing_results_backfilled"] == 1
    assert summary["counts"]["tasks_skipped_existing"] == 1
    preds = json.loads((tmp_path / "out" / "preds.json").read_text())
    assert preds["repo__issue-1__step_0001__action_00__gold__sample_000"]["model_patch"] == "old_patch"


def test_generate_monte_carlo_rollouts_prefers_results_index_over_trajectory_backfill(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    existing_record = {
        "task_key": "repo__issue-1::1::0::0",
        "instance_id": "repo__issue-1",
        "step_index": 1,
        "action_index": 0,
        "sample_index": 0,
        "action_label": "gold",
        "is_gold": True,
        "submission": "indexed_patch",
        "rollout_exit_status": "Submitted",
        "replay_status": "ok",
        "trajectory_path": str(tmp_path / "out" / "indexed.traj.json"),
        "error": None,
    }
    _write_placeholder_file(Path(existing_record["trajectory_path"]))
    _write_existing_results(tmp_path / "out" / "results.jsonl", [existing_record])
    _write_existing_trajectory(
        tmp_path / "out" / "repo__issue-1" / "step_0001" / "gold__sample_000.traj.json",
        submission="backfilled_patch",
    )

    monkeypatch.setattr(
        "minisweagent.run.extra.monte_carlo.load_dataset",
        lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}],
    )

    called = {"count": 0}

    def _fake_run_single_rollout(**kwargs):
        called["count"] += 1
        return {
            "task_key": f"repo__issue-1::1::{kwargs['action_index']}::{kwargs['sample_index']}",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": kwargs["action_index"],
            "sample_index": kwargs["sample_index"],
            "action_label": "alt" if kwargs["action_index"] else "gold",
            "is_gold": kwargs["action_index"] == 0,
            "submission": "new_patch",
            "rollout_exit_status": "Submitted",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / f"{kwargs['action_index']}_{kwargs['sample_index']}.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        show_progress=False,
    )

    assert called["count"] == 1
    assert summary["counts"]["existing_results_backfilled"] == 0
    preds = json.loads((tmp_path / "out" / "preds.json").read_text())
    assert preds["repo__issue-1__step_0001__action_00__gold__sample_000"]["model_patch"] == "indexed_patch"


def test_generate_monte_carlo_rollouts_passes_progress_manager_when_enabled(tmp_path, monkeypatch):
    row = _make_row()
    input_jsonl = tmp_path / "merged.jsonl"
    input_jsonl.write_text(json.dumps(row) + "\n")
    config_path = tmp_path / "mc.yaml"
    _write_config(config_path, tmp_path)

    monkeypatch.setattr(
        "minisweagent.run.extra.monte_carlo.load_dataset",
        lambda *args, **kwargs: [{"instance_id": "repo__issue-1", "problem_statement": "Fix the issue"}],
    )

    observed = {"progress_manager": None, "live_used": False}

    class _FakeProgressManager:
        def __init__(self, *args, **kwargs):
            self.render_group = object()

    class _FakeLive:
        def __init__(self, *args, **kwargs):
            observed["live_used"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_run_single_rollout(**kwargs):
        observed["progress_manager"] = kwargs["progress_manager"]
        return {
            "task_key": f"repo__issue-1::1::{kwargs['action_index']}::{kwargs['sample_index']}",
            "instance_id": "repo__issue-1",
            "step_index": 1,
            "action_index": kwargs["action_index"],
            "sample_index": kwargs["sample_index"],
            "action_label": "alt" if kwargs["action_index"] else "gold",
            "is_gold": kwargs["action_index"] == 0,
            "submission": "",
            "rollout_exit_status": "RolloutStepLimitReached",
            "replay_status": "ok",
            "trajectory_path": str(tmp_path / "out" / f"{kwargs['action_index']}.traj.json"),
            "error": None,
        }

    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.RunBatchProgressManager", _FakeProgressManager)
    monkeypatch.setattr("minisweagent.run.extra.monte_carlo.Live", _FakeLive)
    monkeypatch.setattr("minisweagent.run.extra.monte_carlo._run_single_rollout", _fake_run_single_rollout)

    summary = generate_monte_carlo_rollouts(
        input_jsonl=input_jsonl,
        subset="verified",
        split="dev",
        config_specs=[str(config_path)],
        output_dir=tmp_path / "out",
        samples_per_action=1,
        max_rollout_steps=1,
        max_workers=1,
        show_progress=True,
    )

    assert observed["progress_manager"] is not None
    assert observed["live_used"] is True
    assert summary["counts"]["tasks_completed"] == 2


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


def test_monte_carlo_cli_passes_redo_exit_status(monkeypatch, tmp_path):
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
            "--redo-exit-status",
            "BadRequestError",
            "--redo-exit-status",
            "InternalServerError",
            "--no-show-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["redo_exit_status"] == ["BadRequestError", "InternalServerError"]

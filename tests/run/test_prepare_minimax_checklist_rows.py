import json
from pathlib import Path

from minisweagent.run.utilities.prepare_minimax_checklist_rows import main, prepare_rows


def _write_run(
    root: Path,
    *,
    seed: int,
    resolved_ids: list[str],
    empty_patch_ids: list[str],
    error_ids: list[str],
    trajectories: dict[str, dict],
) -> None:
    run_dir = root / f"test_minimax_2_5_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    submitted_ids = sorted(trajectories.keys())
    unresolved_ids = sorted(set(submitted_ids) - set(resolved_ids))
    report = {
        "report": {
            "schema_version": 2,
            "completed_ids": submitted_ids,
            "completed_instances": len(submitted_ids),
            "empty_patch_ids": empty_patch_ids,
            "empty_patch_instances": len(empty_patch_ids),
            "error_ids": error_ids,
            "error_instances": len(error_ids),
            "incomplete_ids": [],
            "resolved_ids": resolved_ids,
            "resolved_instances": len(resolved_ids),
            "submitted_ids": submitted_ids,
            "submitted_instances": len(submitted_ids),
            "total_instances": len(submitted_ids),
            "unresolved_ids": unresolved_ids,
            "unresolved_instances": len(unresolved_ids),
        }
    }
    (run_dir / "evaluation_result.json").write_text(json.dumps(report))
    for instance_id, payload in trajectories.items():
        instance_dir = run_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / f"{instance_id}.traj.json").write_text(json.dumps(payload))


def _traj(*, task: str, step_count: int, api_calls: int, submission: str, exit_status: str = "Submitted") -> dict:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": task},
        {"role": "assistant", "content": "inspect"},
        {"role": "user", "content": "result"},
    ]
    return {
        "messages": messages,
        "info": {
            "exit_status": exit_status,
            "submission": submission,
            "model_stats": {
                "step_count": step_count,
                "api_calls": api_calls,
            },
        },
    }


def test_prepare_rows_handles_pass_at_n_and_writes_expected_row_sets(tmp_path):
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    _write_run(
        runs_root,
        seed=0,
        resolved_ids=["instance_always_success", "instance_mixed"],
        empty_patch_ids=[],
        error_ids=[],
        trajectories={
            "instance_always_success": _traj(task="Fix always success", step_count=8, api_calls=11, submission="patch-a"),
            "instance_mixed": _traj(task="Fix mixed", step_count=10, api_calls=14, submission="patch-mixed-good"),
            "instance_always_fail": _traj(task="Fix static failure", step_count=5, api_calls=7, submission=""),
        },
    )
    _write_run(
        runs_root,
        seed=1,
        resolved_ids=["instance_always_success"],
        empty_patch_ids=["instance_mixed"],
        error_ids=[],
        trajectories={
            "instance_always_success": _traj(task="Fix always success", step_count=6, api_calls=9, submission="patch-b"),
            "instance_mixed": _traj(task="Fix mixed", step_count=9, api_calls=10, submission=""),
            "instance_always_fail": _traj(task="Fix static failure", step_count=12, api_calls=13, submission="attempted patch"),
        },
    )
    _write_run(
        runs_root,
        seed=2,
        resolved_ids=["instance_always_success"],
        empty_patch_ids=[],
        error_ids=["instance_always_fail"],
        trajectories={
            "instance_always_success": _traj(task="Fix always success", step_count=7, api_calls=10, submission="patch-c"),
            "instance_mixed": _traj(task="Fix mixed", step_count=11, api_calls=15, submission="attempted mixed patch"),
            "instance_always_fail": _traj(task="Fix static failure", step_count=3, api_calls=5, submission="", exit_status="Error"),
        },
    )

    summary = prepare_rows(
        runs_root=runs_root,
        output_dir=output_dir,
        run_prefix="test_minimax_2_5_",
        seeds=[0, 1, 2],
    )

    assert summary["n"] == 3
    assert summary["resolved_union_count"] == 2
    assert summary["total_instances"] == 3
    assert summary["pass_at_n"] == 2 / 3

    success_rows = [json.loads(line) for line in (output_dir / "success_rows.jsonl").read_text().splitlines()]
    pairwise_rows = [json.loads(line) for line in (output_dir / "pairwise_failure_rows.jsonl").read_text().splitlines()]
    static_rows = [json.loads(line) for line in (output_dir / "static_failure_rows.jsonl").read_text().splitlines()]

    success_by_id = {row["instance_id"]: row for row in success_rows}
    assert success_by_id["instance_always_success"]["seed"] == 1
    assert success_by_id["instance_mixed"]["seed"] == 0

    assert len(pairwise_rows) == 1
    assert pairwise_rows[0]["instance_id"] == "instance_mixed"
    assert pairwise_rows[0]["paired_success_seed"] == 0
    assert "assistant: inspect" in pairwise_rows[0]["compare_trajectory_text"]

    assert len(static_rows) == 1
    assert static_rows[0]["instance_id"] == "instance_always_fail"
    assert static_rows[0]["seed"] == 1


def test_main_supports_explicit_seed_list(tmp_path):
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    _write_run(
        runs_root,
        seed=4,
        resolved_ids=["instance_a"],
        empty_patch_ids=[],
        error_ids=[],
        trajectories={"instance_a": _traj(task="Fix a", step_count=3, api_calls=4, submission="patch")},
    )
    _write_run(
        runs_root,
        seed=7,
        resolved_ids=[],
        empty_patch_ids=[],
        error_ids=[],
        trajectories={"instance_a": _traj(task="Fix a", step_count=6, api_calls=8, submission="")},
    )

    main(
        runs_root=runs_root,
        output_dir=output_dir,
        run_prefix="test_minimax_2_5_",
        seeds="4,7",
        seed_start=0,
        seed_end=15,
    )

    summary = json.loads((output_dir / "pass_at_n_summary.json").read_text())
    assert summary["selected_seeds"] == [4, 7]
    assert summary["n"] == 2

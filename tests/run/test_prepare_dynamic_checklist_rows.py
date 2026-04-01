import json

from minisweagent.run.utilities.prepare_dynamic_checklist_rows import main, prepare_dynamic_rows


def _row(instance_id: str, *, category: str) -> dict:
    steps = [
        [{"role": "assistant", "content": "inspect"}, {"role": "user", "content": "result 1"}],
        [{"role": "assistant", "content": "edit"}, {"role": "user", "content": "result 2"}],
    ]
    messages = [message for step in steps for message in step]
    return {
        "instance_id": instance_id,
        "seed": 3,
        "trajectory_path": f"/tmp/{instance_id}.traj.json",
        "task": f"Fix {instance_id}",
        "messages": messages,
        "steps": steps,
        "all_messages": messages,
        "all_steps": steps,
        "selection_metadata": {"category": category, "step_count": 2},
    }


def test_prepare_dynamic_rows_expands_every_step_including_step_zero(tmp_path):
    success_input = tmp_path / "success_rows.jsonl"
    failure_input = tmp_path / "static_failure_rows.jsonl"
    output_dir = tmp_path / "out"
    success_input.write_text(json.dumps(_row("success_case", category="success")) + "\n")
    failure_input.write_text(json.dumps(_row("failure_case", category="static_failure")) + "\n")

    counts = prepare_dynamic_rows(
        success_input=success_input,
        failure_input=failure_input,
        output_dir=output_dir,
    )

    assert counts["dynamic_success_rows"] == 3
    assert counts["dynamic_failure_rows"] == 3

    success_rows = [json.loads(line) for line in (output_dir / "dynamic_success_rows.jsonl").read_text().splitlines()]
    failure_rows = [json.loads(line) for line in (output_dir / "dynamic_failure_rows.jsonl").read_text().splitlines()]

    assert success_rows[0]["step_index"] == 0
    assert success_rows[0]["trajectory_path"] == "/tmp/success_case.traj.json"
    assert sorted(success_rows[0].keys()) == ["instance_id", "seed", "selection_metadata", "step_index", "task", "trajectory_path"]
    assert success_rows[2]["step_index"] == 2
    assert success_rows[1]["selection_metadata"]["dynamic_source_category"] == "dynamic_success"

    assert failure_rows[0]["selection_metadata"]["dynamic_source_category"] == "dynamic_failure"
    assert failure_rows[0]["selection_metadata"]["dynamic_total_steps"] == 2


def test_main_writes_output_files(tmp_path):
    success_input = tmp_path / "success_rows.jsonl"
    failure_input = tmp_path / "static_failure_rows.jsonl"
    output_dir = tmp_path / "out"
    success_input.write_text(json.dumps(_row("success_case", category="success")) + "\n")
    failure_input.write_text(json.dumps(_row("failure_case", category="static_failure")) + "\n")

    main(
        success_input=success_input,
        failure_input=failure_input,
        output_dir=output_dir,
    )

    assert (output_dir / "dynamic_success_rows.jsonl").exists()
    assert (output_dir / "dynamic_failure_rows.jsonl").exists()

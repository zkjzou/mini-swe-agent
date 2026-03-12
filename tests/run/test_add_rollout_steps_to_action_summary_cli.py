from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.add_rollout_steps_to_action_summary import (
    add_rollout_steps_to_action_summary,
    app,
)


def test_add_rollout_steps_to_action_summary_adds_per_action_average(tmp_path: Path):
    summary = {
        "instance_a": {
            "step_0000": {
                "action_00": {"label": "gold", "total": 2},
                "action_01": {"label": "other", "total": 2},
            }
        }
    }
    results_jsonl = tmp_path / "results.jsonl"
    results_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "instance_a", "step_index": 0, "action_index": 0, "rollout_executed_steps": 10}),
                json.dumps({"instance_id": "instance_a", "step_index": 0, "action_index": 0, "rollout_executed_steps": 14}),
                json.dumps({"instance_id": "instance_a", "step_index": 0, "action_index": 1, "rollout_executed_steps": 7}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    updated = add_rollout_steps_to_action_summary(summary, results_jsonl=results_jsonl)

    assert updated["instance_a"]["step_0000"]["action_00"]["avg_rollout_executed_steps"] == 12.0
    assert updated["instance_a"]["step_0000"]["action_00"]["rollout_samples"] == 2
    assert updated["instance_a"]["step_0000"]["action_01"]["avg_rollout_executed_steps"] == 7.0
    assert updated["instance_a"]["step_0000"]["action_01"]["rollout_samples"] == 1


def test_add_rollout_steps_to_action_summary_cli_writes_output(tmp_path: Path):
    input_json = tmp_path / "input.json"
    results_jsonl = tmp_path / "results.jsonl"
    output_json = tmp_path / "output.json"
    input_json.write_text(
        json.dumps({"instance_a": {"step_0000": {"action_00": {"label": "gold", "total": 1}}}}),
        encoding="utf-8",
    )
    results_jsonl.write_text(
        json.dumps({"instance_id": "instance_a", "step_index": 0, "action_index": 0, "rollout_executed_steps": 9}) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-json",
            str(input_json),
            "--results-jsonl",
            str(results_jsonl),
            "--output-json",
            str(output_json),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["instance_a"]["step_0000"]["action_00"]["avg_rollout_executed_steps"] == 9.0

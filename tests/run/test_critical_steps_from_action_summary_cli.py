from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.critical_steps_from_action_summary import app, build_critical_step_report


def test_build_critical_step_report_marks_only_both_gap_steps():
    payload = {
        "instance_a": {
            "step_0000": {
                "action_00": {
                    "label": "gold",
                    "resolve_rate": 0.9,
                    "avg_rollout_executed_steps": 100.0,
                    "rollout_executed_steps_std": 3.0,
                    "rollout_samples": 5,
                    "rollout_executed_steps_values": [100, 101],
                    "resolved": 5,
                    "unresolved": 0,
                    "error": 0,
                    "total": 5,
                },
                "action_01": {
                    "label": "other",
                    "resolve_rate": 0.2,
                    "avg_rollout_executed_steps": 30.0,
                    "rollout_executed_steps_std": 2.0,
                    "rollout_samples": 5,
                    "rollout_executed_steps_values": [30, 31],
                    "resolved": 1,
                    "unresolved": 4,
                    "error": 0,
                    "total": 5,
                },
            },
            "step_0001": {
                "action_00": {"label": "gold", "resolve_rate": 0.9, "avg_rollout_executed_steps": 80.0},
                "action_01": {"label": "other", "resolve_rate": 0.2, "avg_rollout_executed_steps": 70.0},
            },
            "step_0002": {
                "action_00": {"label": "gold", "resolve_rate": 0.8, "avg_rollout_executed_steps": 100.0},
                "action_01": {"label": "other", "resolve_rate": 0.7, "avg_rollout_executed_steps": 20.0},
            },
        }
    }

    report = build_critical_step_report(payload)

    assert report["n_steps"] == 3
    assert report["n_critical_steps"] == 1
    assert report["steps"][0]["step_key"] == "step_0000"
    assert report["steps"][0]["critical_point"] is True
    assert report["steps"][0]["winner_disagreement"] is True
    assert report["steps"][0]["candidate_actions"][0]["action_key"] == "action_00"
    assert report["steps"][0]["candidate_actions"][1]["label"] == "other"
    assert report["steps"][1]["critical_point"] is False
    assert report["steps"][2]["critical_point"] is False


def test_critical_steps_from_action_summary_cli_writes_output(tmp_path: Path):
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "critical_steps.json"
    input_json.write_text(
        json.dumps(
            {
                "instance_a": {
                    "step_0000": {
                        "action_00": {"label": "gold", "resolve_rate": 1.0, "avg_rollout_executed_steps": 100.0},
                        "action_01": {"label": "other", "resolve_rate": 0.0, "avg_rollout_executed_steps": 20.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["n_critical_steps"] == 1
    assert written["steps"][0]["critical_point"] is True

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.enrich_action_summary import app, enrich_action_summary


def test_enrich_action_summary_adds_step_and_action_metrics():
    payload = {
        "instance_a": {
            "step_0000": {
                "action_00": {"label": "gold", "resolved": 5, "unresolved": 0, "error": 0, "total": 5, "resolve_rate": 1.0},
                "action_01": {"label": "other", "resolved": 0, "unresolved": 5, "error": 0, "total": 5, "resolve_rate": 0.0},
            },
            "step_0001": {
                "action_00": {"label": "gold", "resolved": 3, "unresolved": 2, "error": 0, "total": 5, "resolve_rate": 0.6},
                "action_01": {"label": "other", "resolved": 3, "unresolved": 2, "error": 0, "total": 5, "resolve_rate": 0.6},
            },
        }
    }

    enriched = enrich_action_summary(payload, critical_threshold=0.4)

    instance_summary = enriched["instance_a"]["__instance_summary__"]
    assert instance_summary["n_steps"] == 2
    assert instance_summary["n_critical_steps"] == 1

    step_0 = enriched["instance_a"]["step_0000"]["__step_summary__"]
    assert step_0["n_actions"] == 2
    assert step_0["resolve_rate_std"] == 0.5
    assert step_0["critical_point"] is True
    assert step_0["critical_pair"] == ["action_00", "action_01"]

    action_0 = enriched["instance_a"]["step_0000"]["action_00"]
    assert action_0["n_rollouts"] == 5
    assert action_0["resolve_rate_std"] == 0.0
    assert action_0["outcome_rates"] == {"resolved": 1.0, "unresolved": 0.0, "error": 0.0}

    step_1 = enriched["instance_a"]["step_0001"]["__step_summary__"]
    assert step_1["critical_point"] is False
    assert step_1["max_outcome_total_variation"] == 0.0


def test_enrich_action_summary_cli_writes_output(tmp_path: Path):
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    input_json.write_text(
        json.dumps(
            {
                "instance_a": {
                    "step_0000": {
                        "action_00": {
                            "label": "gold",
                            "resolved": 4,
                            "unresolved": 1,
                            "error": 0,
                            "total": 5,
                            "resolve_rate": 0.8,
                        }
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
    assert written["instance_a"]["__instance_summary__"]["n_steps"] == 1
    assert written["instance_a"]["step_0000"]["action_00"]["n_rollouts"] == 5

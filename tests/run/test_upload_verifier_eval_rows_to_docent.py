import importlib.util
import json
from pathlib import Path

import pytest


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "upload_verifier_eval_rows_to_docent.py"
    )
    spec = importlib.util.spec_from_file_location("upload_verifier_eval_rows_to_docent", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_actions_include_rollout_stats_and_step_level_metadata(tmp_path):
    module = _load_script_module()

    action_summary_path = tmp_path / "action_summary.json"
    action_summary_path.write_text(
        json.dumps(
            {
                "demo__task": {
                    "step_0003": {
                        "action_00": {
                            "label": "gold",
                            "resolve_rate": 0.8,
                            "avg_rollout_executed_steps": 12,
                            "rollout_executed_steps_std": 1.5,
                        },
                        "action_01": {
                            "label": "other",
                            "resolve_rate": 0.8,
                            "avg_rollout_executed_steps": 10,
                            "rollout_executed_steps_std": 4.0,
                        },
                        "action_02": {
                            "label": "third",
                            "resolve_rate": 0.5,
                            "avg_rollout_executed_steps": 30,
                            "rollout_executed_steps_std": 2.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    source_row = {
        "instance_id": "demo__task",
        "step_index": 3,
        "actions": [
            {
                "label": "gold",
                "command": "echo gold",
                "is_gold": True,
                "model_response": {"content": [{"type": "text", "text": "gold thought"}]},
            },
            {
                "label": "other",
                "command": "echo other",
                "is_gold": False,
            },
            {
                "label": "third",
                "command": "echo third",
                "is_gold": False,
            },
        ],
    }
    eval_row = {"instance_id": "demo__task", "step_index": 3, "gold_index": 0, "selected_label": "gold"}

    lookup = module.build_candidate_stats_lookup(action_summary_path)
    content = module.build_candidate_actions_content(eval_row, source_row, lookup)
    metadata = module.build_metadata(eval_row, source_row, lookup)

    assert "Average resolve rate: 0.8" in content
    assert "Resolve rate std: 0.4" in content
    assert "Average rollout steps: 12" in content
    assert "Rollout step std: 1.5" in content
    assert metadata["candidate_actions"][0]["resolve_rate"] == 0.8
    assert metadata["candidate_actions"][0]["resolve_rate_std"] == pytest.approx(0.4)
    assert metadata["candidate_actions"][0]["rollout_executed_steps_std"] == 1.5
    assert metadata["monte_carlo_gold_label"] == "other"
    assert metadata["candidate_resolve_rate_std"] == pytest.approx(0.14142135623730953)
    assert metadata["candidate_avg_rollout_steps_std"] == pytest.approx(8.993825042154695)

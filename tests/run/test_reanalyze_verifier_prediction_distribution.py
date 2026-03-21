from __future__ import annotations

import json
from pathlib import Path

from minisweagent.run.utilities.reanalyze_verifier_prediction_distribution import (
    collect_rank_distribution_rows,
    load_action_rankings,
)


def test_collect_rank_distribution_rows_uses_rollout_ranking_and_tracks_missing_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "instance-1": {
                    "step_0000": {
                        "action_00": {"label": "gold", "resolve_rate": 0.8, "avg_rollout_executed_steps": 10},
                        "action_01": {"label": "gpt5mini", "resolve_rate": 1.0, "avg_rollout_executed_steps": 5},
                        "action_02": {"label": "qwen3_5_instruct", "resolve_rate": 1.0, "avg_rollout_executed_steps": 3},
                        "action_03": {"label": "qwen3_coder", "resolve_rate": 0.5, "avg_rollout_executed_steps": 2},
                        "action_04": {"label": "qwen3_coder_next", "resolve_rate": 0.5, "avg_rollout_executed_steps": 4},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "qwen3_5_35b"
    output_dir.mkdir()
    rows_path = output_dir / "basic_verifier_rows.jsonl"
    rows = [
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "instance_id": "instance-1",
            "step_index": 0,
            "selected_index": 2,
            "selected_label": "qwen3_5_instruct",
            "n_actions": 5,
            "verifier_output": {"raw_index": 3},
        },
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "instance_id": "instance-1",
            "step_index": 0,
            "selected_index": 1,
            "selected_label": "gpt5mini",
            "n_actions": 5,
            "verifier_output": {"raw_index": None},
        },
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "instance_id": "instance-1",
            "step_index": 0,
            "selected_index": 0,
            "selected_label": "gold",
            "n_actions": 5,
            "verifier_output": {"raw_index": 1},
        },
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "instance_id": "instance-1",
            "step_index": 1,
            "selected_index": 0,
            "selected_label": "gold",
            "n_actions": 5,
            "verifier_output": {"raw_index": 1},
        },
    ]
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    rankings = load_action_rankings(summary_path)
    csv_rows = collect_rank_distribution_rows(action_rankings=rankings, output_jsonls=[rows_path])

    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["model"] == "llm"
    assert row["verifier_variant"] == "basic_verifier"
    assert row["rows_evaluated"] == "4"
    assert row["rows_with_action_ranking"] == "3"
    assert row["rows_missing_action_ranking"] == "1"
    assert row["rows_non_parser_failed"] == "2"
    assert row["rows_parser_failed"] == "1"
    assert row["aggregated_score"] == "0.375000"
    assert row["strict_gold_score"] == "0.250000"
    assert row["gold"] == "0.500000"
    assert row["2nd"] == "0.000000"
    assert row["3rd"] == "0.500000"
    assert row["count__gold"] == "1"
    assert row["count__3rd"] == "1"
    assert row["parser_failure_count__2nd"] == "1"
    assert row["parser_failure_fraction__2nd"] == "1.000000"

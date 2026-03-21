from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.evaluate_verifier_actions import app, append_predicted_action_distribution
from minisweagent.run.utilities.reanalyze_verifier_predictions import app as reanalyze_app


def test_append_predicted_action_distribution_writes_wide_rows(tmp_path):
    output_jsonl = tmp_path / "rows.jsonl"
    output_csv = tmp_path / "predicted_action_distribution.csv"
    rows = [
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "selected_label": "qwen3_coder",
            "n_actions": 5,
            "gold_index": 0,
            "verifier_output": {"raw_index": 4, "scores": [0.9, 0.6, 0.4, 0.9, 0.1]},
        },
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "selected_label": "qwen3_coder",
            "n_actions": 5,
            "gold_index": 0,
            "verifier_output": {"raw_index": None, "scores": [0.9, 0.6, 0.4, 0.9, 0.1]},
        },
        {
            "status": "evaluated",
            "verifier_type": "llm",
            "verifier_variant": "basic_verifier",
            "selected_label": "gold",
            "n_actions": 5,
            "gold_index": 0,
            "verifier_output": {"raw_index": 1, "scores": [0.9, 0.9, 0.4, 0.2, 0.1]},
        },
        {
            "status": "evaluated",
            "verifier_type": "reward_model",
            "verifier_variant": "world_reward",
            "selected_label": "gold",
            "gold_index": 0,
            "verifier_output": {"rewards": [1.0, 0.5, 0.1, 0.0, -0.2]},
        },
        {"status": "skipped", "verifier_type": "reward_model", "verifier_variant": "world_reward", "selected_label": "ignored"},
    ]
    output_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    append_predicted_action_distribution(output_jsonl, output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[:7] == [
            "model",
            "verifier_variant",
            "gold",
            "qwen3-coder-next",
            "qwen3.5-27b",
            "gpt5-mini",
            "qwen3-coder-instruct",
        ]
        assert reader.fieldnames[7:16] == [
            "rows_evaluated",
            "rows_non_parser_failed",
            "rows_parser_failed",
            "fraction_parser_failed",
            "gold_pick_score_available_count",
            "gold_pick_score_tie_count",
            "gold_pick_score_tie_fraction",
            "timestamp_utc",
            "output_jsonl",
        ]
        assert reader.fieldnames[16:21] == [
            "count__gold",
            "count__qwen3-coder-next",
            "count__qwen3.5-27b",
            "count__gpt5-mini",
            "count__qwen3-coder-instruct",
        ]
        csv_rows = list(reader)

    assert len(csv_rows) == 2
    basic_row = next(row for row in csv_rows if row["verifier_variant"] == "basic_verifier")
    reward_row = next(row for row in csv_rows if row["verifier_variant"] == "world_reward")

    assert basic_row["model"] == "llm"
    assert basic_row["gold"] == "0.500000"
    assert basic_row["qwen3-coder-instruct"] == "0.500000"
    assert basic_row["qwen3-coder-next"] == ""
    assert basic_row["qwen3.5-27b"] == ""
    assert basic_row["gpt5-mini"] == ""
    assert basic_row["rows_evaluated"] == "3"
    assert basic_row["rows_non_parser_failed"] == "2"
    assert basic_row["rows_parser_failed"] == "1"
    assert basic_row["fraction_parser_failed"] == "0.333333"
    assert basic_row["gold_pick_score_available_count"] == "1"
    assert basic_row["gold_pick_score_tie_count"] == "1"
    assert basic_row["gold_pick_score_tie_fraction"] == "1.000000"
    assert basic_row["count__gold"] == "1"
    assert basic_row["count__qwen3-coder-next"] == ""
    assert basic_row["count__qwen3.5-27b"] == ""
    assert basic_row["count__gpt5-mini"] == ""
    assert basic_row["count__qwen3-coder-instruct"] == "1"
    assert basic_row["parser_failure_count__gold"] == ""
    assert basic_row["parser_failure_fraction__gold"] == ""
    assert basic_row["parser_failure_count__qwen3-coder-instruct"] == "1"
    assert basic_row["parser_failure_fraction__qwen3-coder-instruct"] == "1.000000"
    assert basic_row["output_jsonl"] == str(output_jsonl)

    assert reward_row["model"] == "reward_model"
    assert reward_row["gold"] == "1.000000"
    assert reward_row["qwen3-coder-next"] == ""
    assert reward_row["qwen3.5-27b"] == ""
    assert reward_row["gpt5-mini"] == ""
    assert reward_row["qwen3-coder-instruct"] == ""
    assert reward_row["rows_evaluated"] == "1"
    assert reward_row["rows_non_parser_failed"] == "1"
    assert reward_row["rows_parser_failed"] == "0"
    assert reward_row["fraction_parser_failed"] == "0.000000"
    assert reward_row["gold_pick_score_available_count"] == "1"
    assert reward_row["gold_pick_score_tie_count"] == "0"
    assert reward_row["gold_pick_score_tie_fraction"] == "0.000000"
    assert reward_row["count__gold"] == "1"
    assert reward_row["count__qwen3-coder-next"] == ""
    assert reward_row["count__qwen3.5-27b"] == ""
    assert reward_row["count__gpt5-mini"] == ""
    assert reward_row["count__qwen3-coder-instruct"] == ""
    assert reward_row["parser_failure_count__gold"] == ""
    assert reward_row["parser_failure_fraction__gold"] == ""
    assert reward_row["parser_failure_count__qwen3-coder-instruct"] == ""
    assert reward_row["parser_failure_fraction__qwen3-coder-instruct"] == ""


def test_evaluate_verifier_actions_cli_invokes_utility(monkeypatch, tmp_path):
    called = {}
    appended = {}

    def _fake_evaluate(**kwargs):
        called.update(kwargs)
        return {
            "output_jsonl": str(tmp_path / "rows.jsonl"),
            "output_summary": str(tmp_path / "summary.json"),
            "counts": {"rows_considered": 12, "rows_written": 24, "invalid_rows": 0},
            "overall": {
                "rows_evaluated": 12,
                "gold_pick_count": 9,
                "accuracy": 0.75,
                "rows_skipped": 0,
                "rows_failed": 0,
                "total_cost": 1.25,
                "average_cost": 1.25 / 12,
                "total_api_calls": 18,
            },
            "per_variant": {
                "world_reward": {
                    "rows_evaluated": 12,
                    "gold_pick_count": 9,
                    "accuracy": 0.75,
                    "rows_skipped": 0,
                    "rows_failed": 0,
                    "total_cost": 1.25,
                    "average_cost": 1.25 / 12,
                    "total_api_calls": 18,
                }
            },
            "per_verifier": {
                "llm": {
                    "rows_evaluated": 12,
                    "gold_pick_count": 9,
                    "accuracy": 0.75,
                    "rows_skipped": 0,
                    "rows_failed": 0,
                    "total_cost": 1.25,
                    "average_cost": 1.25 / 12,
                    "total_api_calls": 18,
                }
            },
        }

    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluate_verifier_actions.evaluate_verifier_action_selection",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluate_verifier_actions.append_predicted_action_distribution",
        lambda output_jsonl, output_csv: appended.update({"output_jsonl": output_jsonl, "output_csv": output_csv}),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-jsonl",
            str(tmp_path / "merged.jsonl"),
            "--output-jsonl",
            str(tmp_path / "eval_rows.jsonl"),
            "--output-summary",
            str(tmp_path / "eval_summary.json"),
            "--output-distribution-csv",
            str(tmp_path / "predicted_action_distribution.csv"),
            "-c",
            "swebench.yaml",
            "-c",
            'agent.verifier.model.model_name="fake/verifier"',
            "--verifier-type",
            "llm",
            "--verifier-type",
            "reward_model",
            "--verifier-variant",
            "world_reward",
            "--no-strict-five-actions",
            "--no-show-progress",
            "--max-workers",
            "3",
            "--limit-rows",
            "5",
            "--enable-langfuse",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["input_jsonl"] == Path(tmp_path / "merged.jsonl")
    assert called["output_jsonl"] == Path(tmp_path / "eval_rows.jsonl")
    assert called["output_summary"] == Path(tmp_path / "eval_summary.json")
    assert called["config_specs"] == ["swebench.yaml", 'agent.verifier.model.model_name="fake/verifier"']
    assert called["verifier_types"] == ["llm", "reward_model"]
    assert called["verifier_variants"] == ["world_reward"]
    assert called["strict_five_actions"] is False
    assert called["show_progress"] is False
    assert called["max_workers"] == 3
    assert called["limit_rows"] == 5
    assert called["enable_langfuse"] is True
    assert called["overwrite"] is True
    assert appended["output_jsonl"] == Path(tmp_path / "rows.jsonl")
    assert appended["output_csv"] == Path(tmp_path / "predicted_action_distribution.csv")
    assert "overall: evaluated=12 gold_picks=9 accuracy=0.7500" in result.output
    assert "world_reward: evaluated=12 gold_picks=9 accuracy=0.7500" in result.output
    assert "aggregate[llm]: evaluated=12 gold_picks=9 accuracy=0.7500" in result.output
    assert result.output.count("cost=$1.2500") >= 3
    assert result.output.count("avg_cost=$0.1042") >= 3


def test_evaluate_verifier_actions_cli_returns_error_code_on_failure(monkeypatch, tmp_path):
    def _fake_evaluate(**kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluate_verifier_actions.evaluate_verifier_action_selection",
        _fake_evaluate,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-jsonl",
            str(tmp_path / "merged.jsonl"),
            "--output-jsonl",
            str(tmp_path / "eval_rows.jsonl"),
        ],
    )

    assert result.exit_code == 1


def test_reanalyze_verifier_predictions_cli_writes_combined_csv(tmp_path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    (rows_dir / "world_verifier_rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "status": "evaluated",
                        "verifier_type": "llm",
                        "verifier_variant": "world_verifier",
                        "selected_label": "gold",
                        "n_actions": 5,
                        "verifier_output": {"raw_index": None},
                    }
                ),
                json.dumps(
                    {
                        "status": "evaluated",
                        "verifier_type": "llm",
                        "verifier_variant": "world_verifier",
                        "selected_label": "qwen3_coder",
                        "n_actions": 5,
                        "verifier_output": {"raw_index": 4},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "combined.csv"

    runner = CliRunner()
    result = runner.invoke(
        reanalyze_app,
        [
            "--input-dir",
            str(rows_dir),
            "--output-csv",
            str(output_csv),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["verifier_variant"] == "world_verifier"
    assert row["rows_evaluated"] == "2"
    assert row["rows_non_parser_failed"] == "1"
    assert row["rows_parser_failed"] == "1"
    assert row["fraction_parser_failed"] == "0.500000"
    assert row["gold"] == ""
    assert row["qwen3-coder-instruct"] == "1.000000"
    assert row["count__gold"] == ""
    assert row["count__qwen3-coder-next"] == ""
    assert row["count__qwen3.5-27b"] == ""
    assert row["count__gpt5-mini"] == ""
    assert row["count__qwen3-coder-instruct"] == "1"
    assert row["parser_failure_count__gold"] == "1"
    assert row["parser_failure_fraction__gold"] == "1.000000"

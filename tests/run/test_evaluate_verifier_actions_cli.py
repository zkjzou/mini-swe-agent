from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.evaluate_verifier_actions import app


def test_evaluate_verifier_actions_cli_invokes_utility(monkeypatch, tmp_path):
    called = {}

    def _fake_evaluate(**kwargs):
        called.update(kwargs)
        return {
            "output_jsonl": str(tmp_path / "rows.jsonl"),
            "output_summary": str(tmp_path / "summary.json"),
            "counts": {"rows_considered": 12, "rows_written": 24, "invalid_rows": 0},
            "per_verifier": {
                "llm": {"rows_evaluated": 12, "gold_pick_count": 9, "accuracy": 0.75, "rows_skipped": 0, "rows_failed": 0}
            },
        }

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
            "--output-summary",
            str(tmp_path / "eval_summary.json"),
            "-c",
            "swebench.yaml",
            "-c",
            'agent.verifier.model.model_name="fake/verifier"',
            "--verifier-type",
            "llm",
            "--verifier-type",
            "reward_model",
            "--no-strict-five-actions",
            "--no-show-progress",
            "--limit-rows",
            "5",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["input_jsonl"] == Path(tmp_path / "merged.jsonl")
    assert called["output_jsonl"] == Path(tmp_path / "eval_rows.jsonl")
    assert called["output_summary"] == Path(tmp_path / "eval_summary.json")
    assert called["config_specs"] == ["swebench.yaml", 'agent.verifier.model.model_name="fake/verifier"']
    assert called["verifier_types"] == ["llm", "reward_model"]
    assert called["strict_five_actions"] is False
    assert called["show_progress"] is False
    assert called["limit_rows"] == 5
    assert called["overwrite"] is True


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

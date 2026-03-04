from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.merge_verifier_actions import app


def test_merge_verifier_actions_cli_invokes_merge(monkeypatch, tmp_path):
    called = {}

    def _fake_merge(**kwargs):
        called.update(kwargs)
        return {"output_jsonl": str(tmp_path / "merged.jsonl"), "counts": {"rows_kept": 42}}

    monkeypatch.setattr("minisweagent.run.utilities.merge_verifier_actions.merge_verifier_sampling_datasets", _fake_merge)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inputs",
            str(tmp_path / "a.jsonl"),
            "--inputs",
            str(tmp_path / "b.jsonl"),
            "--output-jsonl",
            str(tmp_path / "merged.jsonl"),
            "--output-summary",
            str(tmp_path / "summary.json"),
            "--dedupe",
            "exact",
            "--conflict-policy",
            "keep_last",
            "--no-require-gold",
            "--no-sort-rows",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["input_paths"] == [Path(tmp_path / "a.jsonl"), Path(tmp_path / "b.jsonl")]
    assert called["output_jsonl"] == Path(tmp_path / "merged.jsonl")
    assert called["output_summary"] == Path(tmp_path / "summary.json")
    assert called["dedupe"] == "exact"
    assert called["conflict_policy"] == "keep_last"
    assert called["require_gold"] is False
    assert called["sort_by_instance_step_model_sample"] is False
    assert called["overwrite"] is True


def test_merge_verifier_actions_cli_returns_error_code_on_failure(monkeypatch, tmp_path):
    def _fake_merge(**kwargs):
        raise RuntimeError("merge failed")

    monkeypatch.setattr("minisweagent.run.utilities.merge_verifier_actions.merge_verifier_sampling_datasets", _fake_merge)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inputs",
            str(tmp_path / "a.jsonl"),
            "--output-jsonl",
            str(tmp_path / "merged.jsonl"),
        ],
    )
    assert result.exit_code == 1

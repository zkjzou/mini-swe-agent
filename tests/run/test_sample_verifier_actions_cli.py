from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.sample_verifier_actions import app


def test_sample_verifier_actions_cli_invokes_generator(monkeypatch, tmp_path):
    called = {}

    def _fake_generate(**kwargs):
        called.update(kwargs)
        return {
            "output_jsonl": str(tmp_path / "out" / "candidates.jsonl"),
            "counts": {
                "runs_processed": 1,
                "steps_processed": 2,
                "gold_candidates": 2,
                "sample_candidates": 4,
                "sample_failures": 0,
            },
        }

    monkeypatch.setattr("minisweagent.run.utilities.sample_verifier_actions.generate_verifier_sampling_dataset", _fake_generate)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--output-json",
            str(tmp_path / "output.json"),
            "--transcripts-dir",
            str(tmp_path / "transcripts"),
            "--sampler-config",
            str(tmp_path / "samplers.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-samples",
            "3",
            "--max-workers",
            "5",
            "--limit-runs",
            "10",
            "--limit-steps-per-run",
            "8",
            "--resample-invalid-only",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["output_json_path"] == Path(tmp_path / "output.json")
    assert called["num_samples"] == 3
    assert called["max_workers"] == 5
    assert called["limit_runs"] == 10
    assert called["limit_steps_per_run"] == 8
    assert called["resample_invalid_only"] is True
    assert called["overwrite"] is True


def test_sample_verifier_actions_cli_returns_error_code_on_failure(monkeypatch, tmp_path):
    def _fake_generate(**kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr("minisweagent.run.utilities.sample_verifier_actions.generate_verifier_sampling_dataset", _fake_generate)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--output-json",
            str(tmp_path / "output.json"),
            "--transcripts-dir",
            str(tmp_path / "transcripts"),
            "--sampler-config",
            str(tmp_path / "samplers.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minisweagent.run.utilities.evaluation_client import (
    FINAL_RESULT_FILENAME,
    STAGED_REPORT_FILENAME,
    SUBMISSION_METADATA_FILENAME,
    EvaluationSubmissionMetadata,
    app,
    auto_submit_swebench_predictions,
    derive_rerun_run_id,
    derive_stable_run_id,
    load_submission_metadata,
    make_unique_predictions_upload_copy,
    sync_evaluation_result,
)


def test_derive_stable_run_id_is_stable(tmp_path):
    run_id_a = derive_stable_run_id(output_path=tmp_path / "run-a", subset="swe-bench_verified", split="test")
    run_id_b = derive_stable_run_id(output_path=tmp_path / "run-a", subset="swe-bench_verified", split="test")
    run_id_c = derive_stable_run_id(output_path=tmp_path / "run-b", subset="swe-bench_verified", split="test")

    assert run_id_a == run_id_b
    assert run_id_a != run_id_c


def test_derive_rerun_run_id_appends_timestamp():
    assert derive_rerun_run_id("run-123", created_at=1700000000) == "run-123-rerun-1700000000"


def test_make_unique_predictions_upload_copy_preserves_preds_json(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text('{"instance": {"model_patch": "diff"}}', encoding="utf-8")

    upload_path = make_unique_predictions_upload_copy(
        preds_path,
        output_path=tmp_path,
        subset="swe-bench_verified",
        split="test",
        created_at=1700000000,
    )

    assert preds_path.exists()
    assert upload_path.exists()
    assert upload_path.name.endswith("-swe-bench_verified-test-1700000000.preds.json")
    assert upload_path.read_text(encoding="utf-8") == preds_path.read_text(encoding="utf-8")


def test_auto_submit_swebench_predictions_writes_metadata(monkeypatch, tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text('{"instance": {"model_patch": "diff"}}', encoding="utf-8")

    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.time.time", lambda: 1700000000)
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.submit_predictions_file",
        lambda *args, **kwargs: {
            "job_id": "job-123",
            "run_id": kwargs["run_id"],
            "status": "queued",
            "position_in_queue": 1,
        },
    )

    response, upload_path, metadata_path = auto_submit_swebench_predictions(
        preds_path=preds_path,
        output_dir=tmp_path,
        subset="swe-bench_verified",
        split="test",
        server_url="http://server:8000",
    )

    metadata = load_submission_metadata(metadata_path)
    assert response["job_id"] == "job-123"
    assert upload_path.exists()
    assert metadata.job_id == "job-123"
    assert metadata.server_url == "http://server:8000"
    assert metadata.upload_path == str(upload_path)
    assert metadata.run_id.startswith(tmp_path.name)


def test_auto_submit_swebench_predictions_can_force_rerun(monkeypatch, tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text('{"instance": {"model_patch": "diff"}}', encoding="utf-8")

    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.time.time", lambda: 1700000000)
    called = {}

    def _fake_submit(predictions_path: Path, **kwargs):
        called["predictions_path"] = predictions_path
        called.update(kwargs)
        return {
            "job_id": "job-456",
            "run_id": kwargs["run_id"],
            "status": "queued",
            "position_in_queue": 3,
        }

    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.submit_predictions_file", _fake_submit)

    _, _, metadata_path = auto_submit_swebench_predictions(
        preds_path=preds_path,
        output_dir=tmp_path,
        subset="swe-bench_verified",
        split="test",
        server_url="http://server:8000",
        rerun=True,
    )

    metadata = load_submission_metadata(metadata_path)
    assert called["run_id"].endswith("-rerun-1700000000")
    assert metadata.run_id == called["run_id"]


def test_submit_preds_cli_invokes_submit_and_writes_metadata(monkeypatch, tmp_path):
    called = {}

    def _fake_submit(predictions_path: Path, **kwargs):
        called["predictions_path"] = predictions_path
        called.update(kwargs)
        return {"job_id": "job-123", "run_id": "run-123", "status": "queued", "position_in_queue": 2}

    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.submit_predictions_file", _fake_submit)
    preds_path = tmp_path / "preds.json"
    preds_path.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "submit-preds",
            str(preds_path),
            "--server-url",
            "http://server:8000",
            "--subset",
            "swe-bench_verified",
            "--split",
            "test",
            "--run-id",
            "run-123",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert called["predictions_path"] == preds_path
    metadata = json.loads((tmp_path / SUBMISSION_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["job_id"] == "job-123"
    assert metadata["run_id"] == "run-123"


def test_submit_preds_cli_rerun_rewrites_run_id(monkeypatch, tmp_path):
    called = {}

    def _fake_submit(predictions_path: Path, **kwargs):
        called["predictions_path"] = predictions_path
        called.update(kwargs)
        return {"job_id": "job-123", "run_id": kwargs["run_id"], "status": "queued", "position_in_queue": 2}

    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.submit_predictions_file", _fake_submit)
    monkeypatch.setattr("minisweagent.run.utilities.evaluation_client.time.time", lambda: 1700000000)
    preds_path = tmp_path / "preds.json"
    preds_path.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "submit-preds",
            str(preds_path),
            "--server-url",
            "http://server:8000",
            "--subset",
            "swe-bench_verified",
            "--split",
            "test",
            "--run-id",
            "run-123",
            "--rerun",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert called["run_id"] == "run-123-rerun-1700000000"
    metadata = json.loads((tmp_path / SUBMISSION_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-123-rerun-1700000000"


def test_submit_instance_cli_uses_staged_flow(monkeypatch, tmp_path):
    submit_called = {}
    start_called = {}

    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.submit_single_prediction",
        lambda **kwargs: submit_called.update(kwargs) or {"launched": True, "instance_id": kwargs["instance_id"]},
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.start_staged_run",
        lambda **kwargs: start_called.update(kwargs) or {"started": True, "run_id": kwargs["run_id"]},
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "submit-instance",
            "--instance-id",
            "sympy__sympy-1",
            "--model-name",
            "model-x",
            "--model-patch",
            "diff --git a/a b/a\n",
            "--run-id",
            "run-123",
            "--subset",
            "swe-bench_verified",
            "--split",
            "test",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert submit_called["instance_id"] == "sympy__sympy-1"
    assert start_called["run_id"] == "run-123"
    metadata = json.loads((tmp_path / SUBMISSION_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["source"] == "staged"
    assert metadata["instance_id"] == "sympy__sympy-1"


def test_sync_evaluation_result_writes_final_job_result(monkeypatch, tmp_path):
    metadata = EvaluationSubmissionMetadata(
        source="evaluations",
        server_url="http://server:8000",
        subset="swe-bench_verified",
        split="test",
        run_id="run-123",
        created_at=1.0,
        job_id="job-123",
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.get_evaluation_job",
        lambda **kwargs: {"job_id": "job-123", "status": "succeeded", "position_in_queue": 0},
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.get_evaluation_result",
        lambda **kwargs: {"job_id": "job-123", "report": {"resolved_instances": 1}},
    )

    result = sync_evaluation_result(
        server_url="http://server:8000",
        output_dir=tmp_path,
        metadata=metadata,
        wait=False,
    )

    assert result["status"] == "succeeded"
    assert (tmp_path / FINAL_RESULT_FILENAME).exists()
    assert json.loads((tmp_path / FINAL_RESULT_FILENAME).read_text(encoding="utf-8"))["job_id"] == "job-123"
    saved_metadata = json.loads((tmp_path / SUBMISSION_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert saved_metadata["status"] == "succeeded"


def test_sync_evaluation_result_writes_staged_report(monkeypatch, tmp_path):
    metadata = EvaluationSubmissionMetadata(
        source="staged",
        server_url="http://server:8000",
        subset="swe-bench_verified",
        split="test",
        run_id="run-123",
        created_at=1.0,
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.poll_staged_run",
        lambda **kwargs: {"run_id": "run-123", "pending": [], "running": [], "completed": ["sympy__sympy-1"]},
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.evaluation_client.get_staged_report",
        lambda **kwargs: {"run_id": "run-123", "report": {"resolved_instances": 1}},
    )

    result = sync_evaluation_result(
        server_url="http://server:8000",
        output_dir=tmp_path,
        metadata=metadata,
        wait=False,
    )

    assert result["status"] == "completed"
    assert (tmp_path / STAGED_REPORT_FILENAME).exists()
    assert json.loads((tmp_path / STAGED_REPORT_FILENAME).read_text(encoding="utf-8"))["run_id"] == "run-123"

#!/usr/bin/env python3

"""Client utilities for the local/remote SWE-bench evaluation server."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console

DEFAULT_EVAL_SERVER_URL = "http://laplace.eecs.umich.edu:8000"
SUBMISSION_METADATA_FILENAME = "evaluation_submission.json"
FINAL_RESULT_FILENAME = "evaluation_result.json"
STAGED_REPORT_FILENAME = "evaluation_staged_report.json"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


@dataclass
class EvaluationSubmissionMetadata:
    source: str
    server_url: str
    subset: str
    split: str
    run_id: str
    created_at: float
    predictions_path: str | None = None
    upload_path: str | None = None
    job_id: str | None = None
    status: str | None = None
    position_in_queue: int | None = None
    instance_id: str | None = None
    launched: bool | None = None
    start_response: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


def _normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def _safe_path_stem(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "run"


def derive_stable_run_id(*, output_path: Path, subset: str, split: str) -> str:
    output_path = output_path.resolve()
    digest = sha1(str(output_path).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_path_stem(output_path.name)}-{subset}-{split}-{digest}"


def derive_rerun_run_id(base_run_id: str, *, created_at: float | None = None) -> str:
    timestamp = int(created_at or time.time())
    return f"{base_run_id}-rerun-{timestamp}"


def make_unique_predictions_upload_copy(
    preds_path: Path,
    *,
    output_path: Path | None = None,
    subset: str,
    split: str,
    created_at: float | None = None,
) -> Path:
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    created_at = created_at or time.time()
    output_dir = output_path or preds_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_path_stem((output_path or preds_path.parent).name or preds_path.stem)
    upload_name = f"{base_name}-{subset}-{split}-{int(created_at)}.preds.json"
    upload_path = output_dir / upload_name
    shutil.copy2(preds_path, upload_path)
    return upload_path


def write_submission_metadata(metadata: EvaluationSubmissionMetadata, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / SUBMISSION_METADATA_FILENAME
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata_path


def load_submission_metadata(metadata_path: Path) -> EvaluationSubmissionMetadata:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return EvaluationSubmissionMetadata(**payload)


def _post_json(server_url: str, endpoint: str, payload: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    response = requests.post(
        f"{_normalize_server_url(server_url)}{endpoint}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _get_json(server_url: str, endpoint: str, *, params: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(
        f"{_normalize_server_url(server_url)}{endpoint}",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def submit_predictions_file(
    predictions_path: Path,
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    subset: str,
    split: str,
    run_id: str | None = None,
    timeout: int | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"subset": subset, "split": split}
    if run_id:
        data["run_id"] = run_id
    if timeout is not None:
        data["timeout"] = timeout
    if max_workers is not None:
        data["max_workers"] = max_workers
    with predictions_path.open("rb") as handle:
        response = requests.post(
            f"{_normalize_server_url(server_url)}/evaluations",
            data=data,
            files={"predictions_file": (predictions_path.name, handle, "application/json")},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()


def submit_single_prediction(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    run_id: str,
    subset: str,
    split: str,
    instance_id: str,
    model_name_or_path: str,
    model_patch: str,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "subset": subset,
        "split": split,
        "prediction": {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
        },
    }
    return _post_json(server_url, "/submit", payload)


def start_staged_run(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    run_id: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    return _post_json(server_url, "/runs/start", {"run_id": run_id, "subset": subset, "split": split})


def get_evaluation_job(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    job_id: str,
) -> dict[str, Any]:
    return _get_json(server_url, f"/evaluations/{job_id}")


def get_evaluation_result(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    job_id: str,
) -> dict[str, Any]:
    return _get_json(server_url, f"/evaluations/{job_id}/result")


def poll_staged_run(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    run_id: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    return _get_json(
        server_url,
        "/poll-jobs",
        params={"run_id": run_id, "subset": subset, "split": split},
    )


def get_staged_report(
    *,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    run_id: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    return _post_json(server_url, "/get-report", {"run_id": run_id, "subset": subset, "split": split})


def sync_evaluation_result(
    *,
    server_url: str,
    output_dir: Path,
    metadata: EvaluationSubmissionMetadata,
    wait: bool = False,
    poll_interval: float = 10.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata.source == "evaluations":
        if not metadata.job_id:
            raise ValueError("Job metadata is missing job_id.")
        while True:
            job = get_evaluation_job(server_url=server_url, job_id=metadata.job_id)
            updated = EvaluationSubmissionMetadata(**(asdict(metadata) | {
                "status": job.get("status"),
                "position_in_queue": job.get("position_in_queue"),
                "response": job,
            }))
            write_submission_metadata(updated, output_dir)
            if job.get("status") == "succeeded":
                result = get_evaluation_result(server_url=server_url, job_id=metadata.job_id)
                result_path = output_dir / FINAL_RESULT_FILENAME
                result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                return {"status": "succeeded", "job": job, "result": result, "result_path": str(result_path)}
            if job.get("status") in {"failed", "interrupted"} or not wait:
                return {"status": job.get("status"), "job": job}
            time.sleep(poll_interval)

    while True:
        poll = poll_staged_run(
            server_url=server_url,
            run_id=metadata.run_id,
            subset=metadata.subset,
            split=metadata.split,
        )
        updated = EvaluationSubmissionMetadata(**(asdict(metadata) | {"response": poll}))
        write_submission_metadata(updated, output_dir)
        pending = list(poll.get("pending") or []) + list(poll.get("running") or [])
        if not pending:
            report = get_staged_report(
                server_url=server_url,
                run_id=metadata.run_id,
                subset=metadata.subset,
                split=metadata.split,
            )
            report_path = output_dir / STAGED_REPORT_FILENAME
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return {"status": "completed", "poll": poll, "result": report, "result_path": str(report_path)}
        if not wait:
            return {"status": "running", "poll": poll}
        time.sleep(poll_interval)


def auto_submit_swebench_predictions(
    *,
    preds_path: Path,
    output_dir: Path,
    subset: str,
    split: str,
    server_url: str = DEFAULT_EVAL_SERVER_URL,
    run_id: str | None = None,
    rerun: bool = False,
    timeout: int | None = None,
    max_workers: int | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    created_at = time.time()
    base_run_id = run_id or derive_stable_run_id(output_path=output_dir, subset=subset, split=split)
    resolved_run_id = derive_rerun_run_id(base_run_id, created_at=created_at) if rerun else base_run_id
    upload_path = make_unique_predictions_upload_copy(
        preds_path,
        output_path=output_dir,
        subset=subset,
        split=split,
        created_at=created_at,
    )
    response = submit_predictions_file(
        upload_path,
        server_url=server_url,
        subset=subset,
        split=split,
        run_id=resolved_run_id,
        timeout=timeout,
        max_workers=max_workers,
    )
    metadata = EvaluationSubmissionMetadata(
        source="evaluations",
        server_url=server_url,
        subset=subset,
        split=split,
        run_id=response.get("run_id") or resolved_run_id,
        created_at=created_at,
        predictions_path=str(preds_path),
        upload_path=str(upload_path),
        job_id=response.get("job_id"),
        status=response.get("status"),
        position_in_queue=response.get("position_in_queue"),
        response=response,
    )
    metadata_path = write_submission_metadata(metadata, output_dir)
    return response, upload_path, metadata_path


@app.command("submit-preds")
def submit_preds_command(
    predictions_file: str = typer.Argument(..., help="Predictions JSON or JSONL file"),
    subset: str = typer.Option("swe-bench_verified", "--subset", help="Dataset subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    server_url: str = typer.Option(DEFAULT_EVAL_SERVER_URL, "--server-url", help="Evaluation server URL"),
    run_id: str | None = typer.Option(None, "--run-id", help="Stable server-side run identifier"),
    rerun: bool = typer.Option(
        False,
        "--rerun/--no-rerun",
        help="Force a fresh evaluation run_id instead of reusing an existing cached evaluation",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory to save submission metadata"),
    timeout: int | None = typer.Option(None, "--timeout", help="Per-instance evaluation timeout"),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Evaluation worker count"),
) -> None:
    predictions_path = Path(predictions_file)
    created_at = time.time()
    resolved_run_id = derive_rerun_run_id(run_id, created_at=created_at) if rerun and run_id else run_id
    response = submit_predictions_file(
        predictions_path,
        server_url=server_url,
        subset=subset,
        split=split,
        run_id=resolved_run_id,
        timeout=timeout,
        max_workers=max_workers,
    )
    console.print(json.dumps(response, indent=2))
    if output_dir:
        metadata = EvaluationSubmissionMetadata(
            source="evaluations",
            server_url=server_url,
            subset=subset,
            split=split,
            run_id=response.get("run_id") or resolved_run_id or predictions_path.stem,
            created_at=created_at,
            predictions_path=str(predictions_path),
            upload_path=str(predictions_path),
            job_id=response.get("job_id"),
            status=response.get("status"),
            position_in_queue=response.get("position_in_queue"),
            response=response,
        )
        path = write_submission_metadata(metadata, Path(output_dir))
        console.print(f"[green]Wrote submission metadata:[/green] {path}")


@app.command("submit-instance")
def submit_instance_command(
    instance_id: str = typer.Option(..., "--instance-id", help="SWE-bench instance id"),
    model_name_or_path: str = typer.Option(..., "--model-name", help="Model identifier for the prediction"),
    model_patch: str | None = typer.Option(None, "--model-patch", help="Patch content"),
    patch_file: str | None = typer.Option(None, "--patch-file", help="Read patch content from a file"),
    run_id: str = typer.Option(..., "--run-id", help="Stable staged run identifier"),
    subset: str = typer.Option("swe-bench_verified", "--subset", help="Dataset subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    server_url: str = typer.Option(DEFAULT_EVAL_SERVER_URL, "--server-url", help="Evaluation server URL"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory to save submission metadata"),
    start_run: bool = typer.Option(True, "--start-run/--no-start-run", help="Launch the staged run after submit"),
) -> None:
    patch_text = model_patch
    if patch_file:
        patch_text = Path(patch_file).read_text(encoding="utf-8")
    if not patch_text:
        raise typer.BadParameter("Provide --model-patch or --patch-file.")

    response = submit_single_prediction(
        server_url=server_url,
        run_id=run_id,
        subset=subset,
        split=split,
        instance_id=instance_id,
        model_name_or_path=model_name_or_path,
        model_patch=patch_text,
    )
    start_response = None
    if start_run:
        start_response = start_staged_run(server_url=server_url, run_id=run_id, subset=subset, split=split)
    console.print(json.dumps({"submit": response, "start": start_response}, indent=2))
    if output_dir:
        metadata = EvaluationSubmissionMetadata(
            source="staged",
            server_url=server_url,
            subset=subset,
            split=split,
            run_id=run_id,
            created_at=time.time(),
            instance_id=instance_id,
            launched=response.get("launched"),
            start_response=start_response,
            response=response,
        )
        path = write_submission_metadata(metadata, Path(output_dir))
        console.print(f"[green]Wrote submission metadata:[/green] {path}")


@app.command("start-run")
def start_run_command(
    run_id: str = typer.Option(..., "--run-id", help="Stable staged run identifier"),
    subset: str = typer.Option("swe-bench_verified", "--subset", help="Dataset subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    server_url: str = typer.Option(DEFAULT_EVAL_SERVER_URL, "--server-url", help="Evaluation server URL"),
) -> None:
    response = start_staged_run(server_url=server_url, run_id=run_id, subset=subset, split=split)
    console.print(json.dumps(response, indent=2))


@app.command("poll-job")
def poll_job_command(
    job_id: str = typer.Option(..., "--job-id", help="Evaluation job id"),
    server_url: str = typer.Option(DEFAULT_EVAL_SERVER_URL, "--server-url", help="Evaluation server URL"),
    fetch_result: bool = typer.Option(False, "--fetch-result", help="Fetch the final result payload if succeeded"),
) -> None:
    job = get_evaluation_job(server_url=server_url, job_id=job_id)
    payload: dict[str, Any] = {"job": job}
    if fetch_result and job.get("status") == "succeeded":
        payload["result"] = get_evaluation_result(server_url=server_url, job_id=job_id)
    console.print(json.dumps(payload, indent=2))


@app.command("poll-run")
def poll_run_command(
    run_id: str = typer.Option(..., "--run-id", help="Stable staged run identifier"),
    subset: str = typer.Option("swe-bench_verified", "--subset", help="Dataset subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    server_url: str = typer.Option(DEFAULT_EVAL_SERVER_URL, "--server-url", help="Evaluation server URL"),
    fetch_report: bool = typer.Option(False, "--fetch-report", help="Fetch the final staged report if complete"),
) -> None:
    poll = poll_staged_run(server_url=server_url, run_id=run_id, subset=subset, split=split)
    payload: dict[str, Any] = {"poll": poll}
    if fetch_report and not list(poll.get("pending") or []) and not list(poll.get("running") or []):
        payload["report"] = get_staged_report(server_url=server_url, run_id=run_id, subset=subset, split=split)
    console.print(json.dumps(payload, indent=2))


@app.command("sync-result")
def sync_result_command(
    metadata_file: str = typer.Option(..., "--metadata-file", help="Saved evaluation submission metadata JSON"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for synced result artifacts"),
    wait: bool = typer.Option(False, "--wait", help="Poll until the remote result is complete"),
    poll_interval: float = typer.Option(10.0, "--poll-interval", help="Polling interval in seconds"),
) -> None:
    metadata_path = Path(metadata_file)
    metadata = load_submission_metadata(metadata_path)
    target_dir = Path(output_dir) if output_dir else metadata_path.parent
    result = sync_evaluation_result(
        server_url=metadata.server_url,
        output_dir=target_dir,
        metadata=metadata,
        wait=wait,
        poll_interval=poll_interval,
    )
    console.print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()

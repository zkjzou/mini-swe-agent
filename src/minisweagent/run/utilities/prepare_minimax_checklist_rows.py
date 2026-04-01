from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)

DEFAULT_RUNS_ROOT = Path("/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "minimax_checklist"
DEFAULT_RUN_PREFIX = "test_minimax_2_5_"


def _parse_seed_spec(seeds: str, seed_start: int, seed_end: int) -> list[int]:
    if seeds.strip():
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--seeds was provided but no valid integers were found.")
        return parsed
    if seed_end < seed_start:
        raise ValueError("--seed-end must be greater than or equal to --seed-start.")
    return list(range(seed_start, seed_end + 1))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _extract_task_from_user_message(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", text, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match is not None else text
    extracted = re.sub(
        r"^\s*Consider the following PR description:\s*",
        "",
        extracted,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return extracted or text


def _extract_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    raise ValueError("Trajectory payload must contain a 'messages' list.")


def _extract_task(payload: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    info = payload.get("info")
    if isinstance(info, dict):
        task = info.get("task")
        if isinstance(task, str) and task.strip():
            return _extract_task_from_user_message(task)
    for key in ("task", "problem_statement"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _extract_task_from_user_message(value)
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return _extract_task_from_user_message(content)
    return ""


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    relevant_messages = messages[2:] if len(messages) >= 2 and messages[0].get("role") == "system" else messages
    steps: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in relevant_messages:
        if message.get("role") == "assistant" and current:
            steps.append(current)
            current = [message]
            continue
        current.append(message)
    if current:
        steps.append(current)
    return steps


def _trajectory_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in messages)


def _submission_length(payload: dict[str, Any]) -> int:
    info = payload.get("info", {})
    if isinstance(info, dict):
        submission = info.get("submission")
        if isinstance(submission, str):
            return len(submission)
    return 0


def _model_stats(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("info", {})
    if isinstance(info, dict):
        stats = info.get("model_stats")
        if isinstance(stats, dict):
            return stats
    return {}


def _safe_int(value: Any, default: int) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _success_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        candidate["step_count"],
        candidate["api_calls"],
        candidate["message_count"],
        candidate["submission_length"],
        candidate["seed"],
    )


def _pairwise_failure_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    return (
        0 if candidate["submitted_patch"] else 1,
        candidate["step_count"] if candidate["submitted_patch"] else 10**9,
        candidate["api_calls"] if candidate["submitted_patch"] else 10**9,
        candidate["message_count"] if candidate["submitted_patch"] else 10**9,
        candidate["submission_length"] if candidate["submitted_patch"] else 10**9,
        candidate["seed"],
    )


def _static_failure_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        0 if candidate["submitted_patch"] else 1,
        -candidate["step_count"],
        -candidate["message_count"],
        -candidate["submission_length"],
        candidate["seed"],
    )


def _build_row(instance_id: str, candidate: dict[str, Any], success_frequency: int, *, category: str) -> dict[str, Any]:
    payload = candidate["payload"]
    messages = candidate["messages"]
    steps = candidate["steps"]
    return {
        "instance_id": instance_id,
        "seed": candidate["seed"],
        "trajectory_path": str(candidate["trajectory_path"]),
        "task": candidate["task"],
        "messages": messages,
        "steps": steps,
        "all_messages": messages,
        "all_steps": steps,
        "selection_metadata": {
            "category": category,
            "success_frequency": success_frequency,
            "step_count": candidate["step_count"],
            "api_calls": candidate["api_calls"],
            "message_count": candidate["message_count"],
            "submission_length": candidate["submission_length"],
            "exit_status": payload.get("info", {}).get("exit_status"),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_rows(
    *,
    runs_root: Path,
    output_dir: Path,
    run_prefix: str,
    seeds: list[int],
) -> dict[str, Any]:
    reports_by_seed: dict[int, dict[str, Any]] = {}
    success_ids_by_seed: dict[int, set[str]] = {}
    empty_patch_ids_by_seed: dict[int, set[str]] = {}
    error_ids_by_seed: dict[int, set[str]] = {}
    all_instances: set[str] = set()
    candidates_by_instance: dict[str, list[dict[str, Any]]] = {}

    for seed in seeds:
        run_dir = runs_root / f"{run_prefix}{seed}"
        report_payload = _load_json(run_dir / "evaluation_result.json")
        report = report_payload.get("report")
        if not isinstance(report, dict):
            raise ValueError(f"Missing report payload in {run_dir / 'evaluation_result.json'}")
        reports_by_seed[seed] = report
        success_ids_by_seed[seed] = set(report.get("resolved_ids", []))
        empty_patch_ids_by_seed[seed] = set(report.get("empty_patch_ids", []))
        error_ids_by_seed[seed] = set(report.get("error_ids", []))
        all_instances.update(report.get("submitted_ids", []))
        all_instances.update(report.get("resolved_ids", []))
        all_instances.update(report.get("unresolved_ids", []))

        for instance_id in report.get("submitted_ids", []):
            trajectory_path = run_dir / instance_id / f"{instance_id}.traj.json"
            if not trajectory_path.exists():
                continue
            payload = _load_json(trajectory_path)
            messages = _extract_messages(payload)
            steps = _messages_to_steps(messages)
            stats = _model_stats(payload)
            candidates_by_instance.setdefault(instance_id, []).append(
                {
                    "seed": seed,
                    "trajectory_path": trajectory_path,
                    "payload": payload,
                    "messages": messages,
                    "steps": steps,
                    "task": _extract_task(payload, messages),
                    "step_count": _safe_int(stats.get("step_count"), len(steps)),
                    "api_calls": _safe_int(stats.get("api_calls"), 10**9),
                    "message_count": len(messages),
                    "submission_length": _submission_length(payload),
                    "submitted_patch": bool(_submission_length(payload)),
                    "is_success": instance_id in success_ids_by_seed[seed],
                    "is_error": instance_id in error_ids_by_seed[seed],
                    "is_empty_patch": instance_id in empty_patch_ids_by_seed[seed],
                }
            )

    resolved_union = sorted(set().union(*success_ids_by_seed.values())) if success_ids_by_seed else []
    success_rows: list[dict[str, Any]] = []
    pairwise_failure_rows: list[dict[str, Any]] = []
    static_failure_rows: list[dict[str, Any]] = []
    per_instance_summary: dict[str, Any] = {}

    for instance_id in sorted(all_instances):
        candidates = candidates_by_instance.get(instance_id, [])
        successes = [candidate for candidate in candidates if candidate["is_success"]]
        failures = [candidate for candidate in candidates if not candidate["is_success"]]
        success_frequency = len(successes)

        per_instance_summary[instance_id] = {
            "success_frequency": success_frequency,
            "failure_frequency": len(failures),
            "resolved": success_frequency > 0,
        }

        if successes:
            best_success = min(successes, key=_success_sort_key)
            success_rows.append(_build_row(instance_id, best_success, success_frequency, category="success"))

        if not failures:
            continue

        pairwise_candidates = [
            candidate for candidate in failures if not candidate["is_error"] and not candidate["is_empty_patch"]
        ]
        static_candidates = [candidate for candidate in failures if not candidate["is_error"]]
        pairwise_pool = pairwise_candidates or failures
        static_pool = static_candidates or failures

        if successes:
            best_pairwise_failure = min(pairwise_pool, key=_pairwise_failure_sort_key)
            pairwise_row = _build_row(
                instance_id,
                best_pairwise_failure,
                success_frequency,
                category="pairwise_failure",
            )
            best_success = min(successes, key=_success_sort_key)
            pairwise_row["paired_success_seed"] = best_success["seed"]
            pairwise_row["paired_success_trajectory_path"] = str(best_success["trajectory_path"])
            pairwise_row["compare_trajectory_text"] = _trajectory_text(best_success["messages"])
            pairwise_failure_rows.append(pairwise_row)
            continue

        best_static_failure = min(static_pool, key=_static_failure_sort_key)
        static_failure_rows.append(
            _build_row(
                instance_id,
                best_static_failure,
                success_frequency,
                category="static_failure",
            )
        )

    summary = {
        "run_prefix": run_prefix,
        "selected_seeds": seeds,
        "n": len(seeds),
        "total_instances": len(all_instances),
        "resolved_union_count": len(resolved_union),
        "pass_at_n": (len(resolved_union) / len(all_instances)) if all_instances else 0.0,
        "per_seed_resolved_counts": {
            str(seed): reports_by_seed[seed].get("resolved_instances", len(success_ids_by_seed[seed])) for seed in seeds
        },
        "per_instance_summary": per_instance_summary,
    }

    _write_json(output_dir / "pass_at_n_summary.json", summary)
    _write_jsonl(output_dir / "success_rows.jsonl", success_rows)
    _write_jsonl(output_dir / "pairwise_failure_rows.jsonl", pairwise_failure_rows)
    _write_jsonl(output_dir / "static_failure_rows.jsonl", static_failure_rows)
    return summary


@app.command()
def main(
    runs_root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--runs-root", exists=True, file_okay=False),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", file_okay=False),
    run_prefix: str = typer.Option(DEFAULT_RUN_PREFIX, "--run-prefix"),
    seeds: str = typer.Option("", "--seeds", help="Comma-separated seed list. Overrides --seed-start/--seed-end."),
    seed_start: int = typer.Option(0, "--seed-start"),
    seed_end: int = typer.Option(15, "--seed-end"),
) -> None:
    selected_seeds = _parse_seed_spec(seeds, seed_start, seed_end)
    summary = prepare_rows(
        runs_root=runs_root,
        output_dir=output_dir,
        run_prefix=run_prefix,
        seeds=selected_seeds,
    )
    typer.echo(
        f"Wrote checklist rows for {summary['n']} seeds to {output_dir} "
        f"(pass@{summary['n']}={summary['resolved_union_count']}/{summary['total_instances']})"
    )


if __name__ == "__main__":
    app()

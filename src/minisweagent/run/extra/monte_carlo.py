#!/usr/bin/env python3

"""Monte Carlo branch rollouts from merged verifier-action rows."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

import typer
from datasets import load_dataset
from rich.console import Console
from rich.live import Live

from minisweagent.agents import get_agent
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    DEFAULT_CONFIG_FILE,
    _resolve_profiled_model_config,
    get_sb_environment,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.extra.utils.trajectory_replay import (
    ReplayError,
    build_candidate_branch_message,
    candidate_label,
    candidate_thought,
    safe_label,
    seed_agent_from_history,
)
from minisweagent.utils.serialize import recursive_merge

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.
"""


def _load_resolved_config(config_specs: list[str]) -> dict[str, Any]:
    specs = [str(spec) for spec in config_specs if str(spec).strip()]
    if not specs:
        specs = [str(DEFAULT_CONFIG_FILE)]
    configs = [get_config_from_spec(spec) for spec in specs]
    merged = recursive_merge(*configs)
    resolved = _resolve_profiled_model_config(merged)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved config must be a mapping.")
    return resolved


def _resolve_instance_lookup(subset: str, split: str) -> dict[str, dict[str, Any]]:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    return {
        str(instance["instance_id"]): instance  # type: ignore[index]
        for instance in load_dataset(dataset_path, split=split)
    }


def _load_rows(
    input_jsonl: Path,
    *,
    instance_filter: set[str] | None,
    step_index: int | None,
    row_start: int | None,
    row_end: int | None,
    limit_rows: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    matched_count = 0
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            instance_id = row.get("instance_id")
            if instance_filter and instance_id not in instance_filter:
                continue
            if step_index is not None and int(row.get("step_index", -1)) != step_index:
                continue
            matched_count += 1
            if row_start is not None and matched_count < row_start:
                continue
            if row_end is not None and matched_count > row_end:
                break
            rows.append((line_no, row))
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    return rows


def _task_key(
    *,
    instance_id: str,
    step_index: int,
    action_index: int,
    sample_index: int,
) -> str:
    return f"{instance_id}::{step_index}::{action_index}::{sample_index}"


def _task_display_id(
    *,
    instance_id: str,
    step_index: int,
    action_label: str,
    sample_index: int,
) -> str:
    return f"{instance_id} s{step_index} {action_label} #{sample_index}"


def _is_terminal(agent: Any) -> bool:
    return bool(agent.messages and isinstance(agent.messages[-1], dict) and agent.messages[-1].get("role") == "exit")


def _get_exit_payload(agent: Any) -> dict[str, Any]:
    if _is_terminal(agent):
        return dict(agent.messages[-1].get("extra", {}) or {})
    return {"exit_status": "RolloutStepLimitReached", "submission": ""}


def _append_exit_message(agent: Any, *, exit_status: str, submission: str = "", content: str | None = None) -> None:
    if _is_terminal(agent):
        return
    agent.add_messages(
        {
            "role": "exit",
            "content": content or exit_status,
            "extra": {
                "exit_status": exit_status,
                "submission": submission,
            },
        }
    )


def _continue_rollout(agent: Any, *, max_rollout_steps: int) -> int:
    while not _is_terminal(agent):
        if agent.step_count >= max_rollout_steps:
            _append_exit_message(agent, exit_status="RolloutStepLimitReached")
            break
        try:
            agent.step()
        except InterruptAgentFlow as exc:
            if isinstance(exc, FormatError) and not agent.config.add_format_error_to_conversation:
                continue
            agent.add_messages(*exc.messages)
        except Exception as exc:  # noqa: BLE001
            agent.handle_uncaught_exception(exc)
            break
    return int(agent.step_count)


def _apply_overrides(
    config: dict[str, Any],
    *,
    model_name: str | None,
    model_class: str | None,
    environment_class: str | None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    model_config = resolved.setdefault("model", {})
    env_config = resolved.setdefault("environment", {})
    if not isinstance(model_config, dict):
        raise ValueError("Resolved model config must be a mapping.")
    if not isinstance(env_config, dict):
        raise ValueError("Resolved environment config must be a mapping.")
    if model_name:
        model_config["model_name"] = model_name
    if model_class:
        model_config["model_class"] = model_class
    if environment_class:
        env_config["environment_class"] = environment_class
    return resolved


def _save_rollout(agent: Any, path: Path, instance_id: str, rollout_info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(
        path,
        {
            "instance_id": instance_id,
            "info": {
                "rollout": rollout_info,
            },
        },
    )


def _install_progress_tracking(agent: Any, progress_manager: RunBatchProgressManager | None, display_id: str) -> None:
    if progress_manager is None or not isinstance(agent, DefaultAgent):
        return

    original_step = agent.step
    agent._mc_display_step = 0

    def step_with_progress(self):
        self._mc_display_step += 1
        progress_manager.update_instance_status(
            display_id,
            f"Step {self._mc_display_step:3d} (${self.cost:.2f})",
        )
        return original_step()

    agent.step = MethodType(step_with_progress, agent)


def _prediction_priority(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    submission = str(record.get("submission") or "")
    exit_status = str(record.get("rollout_exit_status") or "")
    if exit_status == "Submitted" and submission:
        bucket = 0
    elif submission:
        bucket = 1
    else:
        bucket = 2
    return (
        bucket,
        int(record.get("step_index") or 0),
        0 if bool(record.get("is_gold")) else 1,
        int(record.get("action_index") or 0),
        int(record.get("sample_index") or 0),
    )


def _rollout_pred_id(record: dict[str, Any]) -> str | None:
    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        return None
    step_index = int(record.get("step_index") or 0)
    action_index = int(record.get("action_index") or 0)
    action_label = safe_label(str(record.get("action_label") or f"action_{action_index}"))
    sample_index = int(record.get("sample_index") or 0)
    return (
        f"{instance_id}__step_{step_index:04d}__action_{action_index:02d}"
        f"__{action_label}__sample_{sample_index:03d}"
    )


def _load_existing_results(results_path: Path) -> dict[str, dict[str, Any]]:
    if not results_path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                continue
            key = _task_key(
                instance_id=instance_id,
                step_index=int(record.get("step_index") or 0),
                action_index=int(record.get("action_index") or 0),
                sample_index=int(record.get("sample_index") or 0),
            )
            existing[key] = record
    return existing


def _record_from_saved_trajectory(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None

    instance_id = payload.get("instance_id")
    info = payload.get("info")
    if not isinstance(instance_id, str) or not instance_id:
        return None
    if not isinstance(info, dict):
        return None

    rollout = info.get("rollout")
    if not isinstance(rollout, dict):
        return None

    try:
        step_index = int(rollout.get("source_step_index") or 0)
        action_index = int(rollout.get("action_index") or 0)
        sample_index = int(rollout.get("sample_index") or 0)
    except (TypeError, ValueError):
        return None

    record = {
        "task_key": _task_key(
            instance_id=instance_id,
            step_index=step_index,
            action_index=action_index,
            sample_index=sample_index,
        ),
        "line_no": rollout.get("source_jsonl_line"),
        "instance_id": instance_id,
        "step_index": step_index,
        "message_index": rollout.get("source_message_index"),
        "trajectory_relpath": rollout.get("trajectory_relpath"),
        "action_index": action_index,
        "action_label": rollout.get("action_label"),
        "candidate_source": rollout.get("candidate_source"),
        "is_gold": bool(rollout.get("is_gold")),
        "sample_index": sample_index,
        "forced_command": rollout.get("forced_command"),
        "forced_thought": rollout.get("forced_thought"),
        "replay_status": rollout.get("replay_status"),
        "replayed_prefix_steps": rollout.get("replayed_prefix_steps"),
        "rollout_exit_status": rollout.get("rollout_exit_status") or info.get("exit_status"),
        "rollout_executed_steps": rollout.get("rollout_executed_steps"),
        "submission": str(info.get("submission") or ""),
        "cost": rollout.get("cost"),
        "agent_api_calls": rollout.get("agent_api_calls"),
        "verifier_api_calls": rollout.get("verifier_api_calls"),
        "duration_seconds": rollout.get("duration_seconds"),
        "trajectory_path": str(path),
        "error": None,
    }
    return record


def _backfill_existing_results_from_trajectories(
    output_dir: Path,
    existing: dict[str, dict[str, Any]],
) -> int:
    backfilled = 0
    for trajectory_path in sorted(output_dir.rglob("*.traj.json")):
        record = _record_from_saved_trajectory(trajectory_path)
        if record is None:
            continue
        key = record["task_key"]
        if key in existing:
            continue
        existing[key] = record
        backfilled += 1
    return backfilled


def _existing_record_is_error(record: dict[str, Any]) -> bool:
    if record.get("error") not in (None, {}):
        return True
    if record.get("replay_status") == "error":
        return True
    trajectory_path = record.get("trajectory_path")
    if isinstance(trajectory_path, str) and trajectory_path and not Path(trajectory_path).exists():
        return True
    return False


def _existing_record_has_exit_status(record: dict[str, Any], exit_statuses: set[str]) -> bool:
    status = str(record.get("rollout_exit_status") or "").strip()
    return bool(status) and status in exit_statuses


def _write_rollout_preds_file(results: list[dict[str, Any]], *, output_dir: Path, model_name: str) -> Path:
    preds_payload: dict[str, dict[str, Any]] = {}
    for record in sorted(
        results,
        key=lambda item: (
            str(item.get("instance_id") or ""),
            int(item.get("step_index") or 0),
            int(item.get("action_index") or 0),
            int(item.get("sample_index") or 0),
        ),
    ):
        pred_id = _rollout_pred_id(record)
        if pred_id is None:
            continue
        preds_payload[pred_id] = {
            "model_name_or_path": model_name,
            "instance_id": pred_id,
            "model_patch": str(record.get("submission") or ""),
        }

    preds_path = output_dir / "preds.json"
    preds_path.write_text(json.dumps(preds_payload, indent=2, ensure_ascii=False))
    return preds_path


def _write_instance_preds_file(results: list[dict[str, Any]], *, output_dir: Path, model_name: str) -> Path:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            continue
        by_instance.setdefault(instance_id, []).append(record)

    preds_payload: dict[str, dict[str, Any]] = {}
    for instance_id, instance_records in sorted(by_instance.items()):
        selected = sorted(instance_records, key=_prediction_priority)[0]
        preds_payload[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": str(selected.get("submission") or ""),
        }

    preds_path = output_dir / "preds_by_instance.json"
    preds_path.write_text(json.dumps(preds_payload, indent=2, ensure_ascii=False))
    return preds_path


def _run_single_rollout(
    *,
    line_no: int,
    row: dict[str, Any],
    action: dict[str, Any],
    action_index: int,
    sample_index: int,
    config: dict[str, Any],
    instance: dict[str, Any],
    output_dir: Path,
    max_rollout_steps: int,
    progress_manager: RunBatchProgressManager | None = None,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or instance.get("instance_id") or "")
    step = int(row.get("step_index") or 0)
    label = candidate_label(action, action_index)
    display_id = _task_display_id(
        instance_id=instance_id,
        step_index=step,
        action_label=safe_label(label),
        sample_index=sample_index,
    )
    started_at = time.time()
    agent = None
    trajectory_path: Path | None = None

    if progress_manager is not None:
        progress_manager.on_instance_start(display_id)
        progress_manager.update_instance_status(display_id, "Initializing")

    branch_result: dict[str, Any] = {
        "task_key": _task_key(
            instance_id=instance_id,
            step_index=step,
            action_index=action_index,
            sample_index=sample_index,
        ),
        "line_no": line_no,
        "instance_id": instance_id,
        "step_index": step,
        "message_index": row.get("message_index"),
        "run_id": row.get("run_id"),
        "trajectory_relpath": row.get("trajectory_relpath"),
        "action_index": action_index,
        "action_label": label,
        "candidate_source": action.get("candidate_source"),
        "is_gold": bool(action.get("is_gold")),
        "sample_index": sample_index,
        "forced_command": str(action.get("command") or ""),
        "forced_thought": candidate_thought(action),
        "replay_status": "pending",
        "rollout_exit_status": None,
        "submission": "",
        "replayed_prefix_steps": 0,
        "rollout_executed_steps": 0,
        "cost": None,
        "agent_api_calls": None,
        "verifier_api_calls": None,
        "trajectory_path": None,
        "error": None,
    }

    try:
        row_actions = row.get("actions")
        if not isinstance(row_actions, list):
            raise ReplayError("Row is missing an actions list.")
        if len(row_actions) <= action_index:
            raise ReplayError(f"Action index {action_index} is out of range.")

        model = get_model(config=config.get("model", {}))
        env = get_sb_environment(config, instance)
        agent = get_agent(model, env, config.get("agent", {}), default_type="default")
        _install_progress_tracking(agent, progress_manager, display_id)

        if progress_manager is not None:
            progress_manager.update_instance_status(display_id, "Replaying prefix")
        replay_summary = seed_agent_from_history(agent, row)
        branch_result["replay_status"] = "ok"
        branch_result["replayed_prefix_steps"] = replay_summary.replayed_prefix_steps

        branch_message = build_candidate_branch_message(action, action_index=action_index)
        agent.add_messages(branch_message)
        if progress_manager is not None:
            progress_manager.update_instance_status(display_id, "Forced branch")
        try:
            agent.execute_actions(branch_message)
        except InterruptAgentFlow as exc:
            agent.add_messages(*exc.messages)
        except Exception as exc:  # noqa: BLE001
            agent.handle_uncaught_exception(exc)

        if not _is_terminal(agent):
            if progress_manager is not None:
                progress_manager.update_instance_status(display_id, "Continuing rollout")
            branch_result["rollout_executed_steps"] = _continue_rollout(agent, max_rollout_steps=max_rollout_steps)
        else:
            branch_result["rollout_executed_steps"] = 0

        exit_payload = _get_exit_payload(agent)
        branch_result["rollout_exit_status"] = exit_payload.get("exit_status", "Unknown")
        branch_result["submission"] = exit_payload.get("submission", "") or ""
        branch_result["cost"] = getattr(agent, "cost", None)
        branch_result["agent_api_calls"] = getattr(agent, "agent_api_calls", None)
        branch_result["verifier_api_calls"] = getattr(agent, "verifier_api_calls", None)

        rollout_info = {
            "source_jsonl_line": line_no,
            "instance_id": instance_id,
            "source_step_index": step,
            "source_message_index": row.get("message_index"),
            "trajectory_relpath": row.get("trajectory_relpath"),
            "action_index": action_index,
            "action_label": label,
            "candidate_source": action.get("candidate_source"),
            "is_gold": bool(action.get("is_gold")),
            "sample_index": sample_index,
            "forced_command": branch_result["forced_command"],
            "forced_thought": branch_result["forced_thought"],
            "replay_status": branch_result["replay_status"],
            "replayed_prefix_steps": branch_result["replayed_prefix_steps"],
            "rollout_exit_status": branch_result["rollout_exit_status"],
            "rollout_executed_steps": branch_result["rollout_executed_steps"],
            "cost": branch_result["cost"],
            "agent_api_calls": branch_result["agent_api_calls"],
            "verifier_api_calls": branch_result["verifier_api_calls"],
            "duration_seconds": round(time.time() - started_at, 3),
        }
        trajectory_path = (
            output_dir
            / instance_id
            / f"step_{step:04d}"
            / f"{safe_label(label)}__sample_{sample_index:03d}.traj.json"
        )
        _save_rollout(agent, trajectory_path, instance_id, rollout_info)
        branch_result["trajectory_path"] = str(trajectory_path)
        return branch_result
    except Exception as exc:  # noqa: BLE001
        branch_result["replay_status"] = "error" if branch_result["replay_status"] == "pending" else branch_result["replay_status"]
        branch_result["rollout_exit_status"] = type(exc).__name__
        branch_result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        branch_result["trajectory_path"] = str(trajectory_path) if trajectory_path is not None else None
        return branch_result
    finally:
        if progress_manager is not None:
            progress_manager.on_instance_end(display_id, str(branch_result.get("rollout_exit_status")))
        if agent is not None and hasattr(agent, "env"):
            cleanup = getattr(agent.env, "cleanup", None)
            if callable(cleanup):
                cleanup()


def generate_monte_carlo_rollouts(
    *,
    input_jsonl: Path,
    subset: str,
    split: str,
    config_specs: list[str],
    output_dir: Path,
    samples_per_action: int = 1,
    max_rollout_steps: int = 20,
    max_workers: int = 1,
    limit_rows: int | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
    instance_filter: list[str] | None = None,
    step_index: int | None = None,
    model_name: str | None = None,
    model_class: str | None = None,
    environment_class: str | None = None,
    redo_existing: bool = False,
    redo_errors: bool = False,
    redo_exit_status: list[str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    if samples_per_action < 1:
        raise ValueError("samples_per_action must be >= 1")
    if max_rollout_steps < 0:
        raise ValueError("max_rollout_steps must be >= 0")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if row_start is not None and row_start < 1:
        raise ValueError("row_start must be >= 1")
    if row_end is not None and row_end < 1:
        raise ValueError("row_end must be >= 1")
    if row_start is not None and row_end is not None and row_start > row_end:
        raise ValueError("row_start must be <= row_end")

    normalized_redo_exit_status = sorted({str(status).strip() for status in (redo_exit_status or []) if str(status).strip()})
    redo_exit_status_set = set(normalized_redo_exit_status)

    resolved_config = _apply_overrides(
        _load_resolved_config(config_specs),
        model_name=model_name,
        model_class=model_class,
        environment_class=environment_class,
    )
    instance_lookup = _resolve_instance_lookup(subset, split)
    filtered_rows = _load_rows(
        input_jsonl,
        instance_filter=set(instance_filter or []),
        step_index=step_index,
        row_start=row_start,
        row_end=row_end,
        limit_rows=limit_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    existing_results = _load_existing_results(results_path)
    backfilled_existing_results = _backfill_existing_results_from_trajectories(output_dir, existing_results)

    tasks: list[dict[str, Any]] = []
    skipped_rows = Counter()
    skipped_tasks = Counter()
    preserved_records: list[dict[str, Any]] = []
    if redo_existing and (redo_errors or normalized_redo_exit_status):
        override_targets: list[str] = []
        if redo_errors:
            override_targets.append("--redo-errors")
        if normalized_redo_exit_status:
            override_targets.append("--redo-exit-status")
        console.print(f"--redo-existing overrides {' and '.join(override_targets)}; rerunning all rollout tasks.")
    for line_no, row in filtered_rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id not in instance_lookup:
            skipped_rows["missing_instance"] += 1
            continue
        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            skipped_rows["missing_actions"] += 1
            continue
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                skipped_rows["non_mapping_action"] += 1
                continue
            for sample_index in range(samples_per_action):
                task_key = _task_key(
                    instance_id=instance_id,
                    step_index=int(row.get("step_index") or 0),
                    action_index=action_index,
                    sample_index=sample_index,
                )
                existing_record = existing_results.get(task_key)
                if existing_record is not None and not redo_existing:
                    should_redo_for_error = redo_errors and _existing_record_is_error(existing_record)
                    should_redo_for_status = bool(redo_exit_status_set) and _existing_record_has_exit_status(
                        existing_record,
                        redo_exit_status_set,
                    )
                    if redo_errors or normalized_redo_exit_status:
                        if not (should_redo_for_error or should_redo_for_status):
                            preserved_records.append(existing_record)
                            skipped_tasks["existing_non_target"] += 1
                            continue
                    else:
                        preserved_records.append(existing_record)
                        skipped_tasks["existing"] += 1
                        continue
                tasks.append(
                    {
                        "task_key": task_key,
                        "line_no": line_no,
                        "row": row,
                        "action": action,
                        "action_index": action_index,
                        "sample_index": sample_index,
                        "instance": instance_lookup[instance_id],
                    }
                )

    progress_manager = None
    if show_progress:
        progress_manager = RunBatchProgressManager(
            len(tasks),
            output_dir / f"exit_statuses_{time.time()}.yaml",
        )

    results: list[dict[str, Any]] = []
    live_context = Live(progress_manager.render_group, refresh_per_second=4) if progress_manager else nullcontext()
    with live_context:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _run_single_rollout,
                    line_no=task["line_no"],
                    row=task["row"],
                    action=task["action"],
                    action_index=task["action_index"],
                    sample_index=task["sample_index"],
                    config=copy.deepcopy(resolved_config),
                    instance=task["instance"],
                    output_dir=output_dir,
                    max_rollout_steps=max_rollout_steps,
                    progress_manager=progress_manager,
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())

    all_results: list[dict[str, Any]] = preserved_records + results
    all_results = sorted(
        all_results,
        key=lambda record: (
            str(record.get("instance_id") or ""),
            int(record.get("step_index") or 0),
            int(record.get("action_index") or 0),
            int(record.get("sample_index") or 0),
        ),
    )

    with results_path.open("w", encoding="utf-8") as handle:
        for record in all_results:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")

    resolved_model_name = str(((resolved_config.get("model") or {}).get("model_name")) or "")
    preds_path = _write_rollout_preds_file(all_results, output_dir=output_dir, model_name=resolved_model_name)
    preds_by_instance_path = _write_instance_preds_file(
        all_results,
        output_dir=output_dir,
        model_name=resolved_model_name,
    )

    exit_counts = Counter(record.get("rollout_exit_status") or "Unknown" for record in all_results)
    summary = {
        "input_jsonl": str(input_jsonl),
        "subset": subset,
        "split": split,
        "config_specs": list(config_specs) if config_specs else [str(DEFAULT_CONFIG_FILE)],
        "samples_per_action": samples_per_action,
        "max_rollout_steps": max_rollout_steps,
        "max_workers": max_workers,
        "limit_rows": limit_rows,
        "row_start": row_start,
        "row_end": row_end,
        "instance_filter": list(instance_filter or []),
        "step_index": step_index,
        "redo_existing": redo_existing,
        "redo_errors": redo_errors,
        "redo_exit_status": normalized_redo_exit_status,
        "preds_json": str(preds_path),
        "preds_by_instance_json": str(preds_by_instance_path),
        "counts": {
            "rows_loaded": len(filtered_rows),
            "rows_skipped": sum(skipped_rows.values()),
            "tasks_planned": len(tasks),
            "tasks_skipped_existing": sum(skipped_tasks.values()),
            "tasks_completed": len(results),
            "tasks_total_in_results": len(all_results),
            "existing_results_backfilled": backfilled_existing_results,
            "replay_failures": sum(1 for record in all_results if record.get("replay_status") == "error"),
            "solved": exit_counts.get("Submitted", 0),
            "pred_rollouts": len(json.loads(preds_path.read_text())),
            "pred_instances": len(json.loads(preds_by_instance_path.read_text())),
        },
        "skipped_rows": dict(skipped_rows),
        "skipped_tasks": dict(skipped_tasks),
        "exit_status_counts": dict(exit_counts),
        "results_jsonl": str(results_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


# fmt: off
@app.command(help=__doc__)
def main(
    input_jsonl: Path = typer.Argument(..., help="Path to merged_grouped_latest.jsonl"),
    subset: str = typer.Option("verified", "--subset", help="SWE-bench subset or dataset path"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    config_specs: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output_dir: Path = typer.Option(Path("monte_carlo_rollouts"), "--output-dir", help="Directory for rollout outputs"),
    samples_per_action: int = typer.Option(1, "--samples-per-action", min=1, help="Rollouts per candidate action"),
    max_rollout_steps: int = typer.Option(20, "--max-rollout-steps", min=0, help="Continuation steps after the forced branch"),
    max_workers: int = typer.Option(1, "--max-workers", min=1, help="Concurrent rollout workers"),
    limit_rows: int | None = typer.Option(None, "--limit-rows", min=1, help="Optional cap on source rows"),
    row_start: int | None = typer.Option(None, "--row-start", min=1, help="1-based inclusive start row after filters"),
    row_end: int | None = typer.Option(None, "--row-end", min=1, help="1-based inclusive end row after filters"),
    instance: list[str] | None = typer.Option(None, "--instance", help="Optional instance_id filter"),
    step_index: int | None = typer.Option(None, "--step-index", min=0, help="Optional exact step_index filter"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Optional actor model override"),
    model_class: str | None = typer.Option(None, "--model-class", help="Optional actor model class override"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Optional environment class override"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing rollout tasks"),
    redo_errors: bool = typer.Option(False, "--redo-errors", help="Redo only existing rollout tasks whose prior result was an error"),
    redo_exit_status: list[str] | None = typer.Option(
        None,
        "--redo-exit-status",
        help="Redo only existing rollout tasks whose prior rollout_exit_status matches one of these values",
    ),
    show_progress: bool = typer.Option(True, "--show-progress/--no-show-progress", help="Display tqdm progress if available"),
) -> None:
    # fmt: on
    try:
        summary = generate_monte_carlo_rollouts(
            input_jsonl=input_jsonl,
            subset=subset,
            split=split,
            config_specs=config_specs,
            output_dir=output_dir,
            samples_per_action=samples_per_action,
            max_rollout_steps=max_rollout_steps,
            max_workers=max_workers,
            limit_rows=limit_rows,
            row_start=row_start,
            row_end=row_end,
            instance_filter=instance,
            step_index=step_index,
            model_name=model_name,
            model_class=model_class,
            environment_class=environment_class,
            redo_existing=redo_existing,
            redo_errors=redo_errors,
            redo_exit_status=redo_exit_status,
            show_progress=show_progress,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Monte Carlo rollout failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = summary.get("counts", {})
    console.print(f"[green]Wrote:[/green] {summary.get('results_jsonl')}")
    console.print(
        "rows={rows_loaded} planned={tasks_planned} completed={tasks_completed} solved={solved} replay_failures={replay_failures}".format(
            rows_loaded=counts.get("rows_loaded", 0),
            tasks_planned=counts.get("tasks_planned", 0),
            tasks_completed=counts.get("tasks_completed", 0),
            solved=counts.get("solved", 0),
            replay_failures=counts.get("replay_failures", 0),
        )
    )


if __name__ == "__main__":
    app()

from __future__ import annotations

import copy
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from minisweagent.models import get_model

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm may be unavailable in some environments
    _tqdm = None

logger = logging.getLogger(__name__)

DATASET_VERSION = "verifier_candidates_v1"


@dataclass(frozen=True)
class SamplerModelSpec:
    id: str
    model_name: str
    model_config: dict[str, Any]
    sampling_kwargs: dict[str, Any]


@dataclass
class PreparedRun:
    instance_id: str
    run_id: str | None
    problem_id: str | None
    trajectory_relpath: str
    messages: list[dict[str, Any]]
    replay_steps: list[dict[str, Any]]


def load_sampler_model_specs(path: Path) -> list[SamplerModelSpec]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Sampler config must be a mapping, got {type(payload).__name__}.")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Sampler config must define a non-empty 'models' list.")

    specs: list[SamplerModelSpec] = []
    seen_ids: set[str] = set()
    for i, model_entry in enumerate(models):
        if not isinstance(model_entry, dict):
            raise ValueError(f"Model entry {i} must be a mapping.")
        model_id = str(model_entry.get("id", "")).strip()
        model_name = str(model_entry.get("model_name", "")).strip()
        if not model_id:
            raise ValueError(f"Model entry {i} is missing a non-empty 'id'.")
        if not model_name:
            raise ValueError(f"Model entry {i} is missing a non-empty 'model_name'.")
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id '{model_id}' in sampler config.")
        seen_ids.add(model_id)
        sampling_kwargs = model_entry.get("sampling_kwargs") or {}
        if not isinstance(sampling_kwargs, dict):
            raise ValueError(f"Model '{model_id}' has non-mapping 'sampling_kwargs'.")
        model_config = {k: v for k, v in model_entry.items() if k not in {"id", "sampling_kwargs"}}
        specs.append(
            SamplerModelSpec(
                id=model_id,
                model_name=model_name,
                model_config=model_config,
                sampling_kwargs=dict(sampling_kwargs),
            )
        )
    return specs


def _is_resolved_run(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False
    scores = metadata.get("scores") or {}
    if not isinstance(scores, dict):
        return False
    return scores.get("resolved") in (1, True)


def _resolve_transcript_path(transcripts_dir: Path, transcript_ref: str) -> Path:
    ref_path = Path(transcript_ref)
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.append(transcripts_dir / ref_path)
        candidates.append(transcripts_dir / ref_path.name)
        if ref_path.parts and ref_path.parts[0] == "transcripts" and len(ref_path.parts) > 1:
            candidates.append(transcripts_dir.joinpath(*ref_path.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Transcript file not found for '{transcript_ref}'. Tried: {tried}")


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    tool_call_id = tool_call.get("id")
    function_obj = tool_call.get("function")
    raw_arguments = tool_call.get("arguments")
    function_name: str | None = None
    if isinstance(function_obj, dict):
        function_name = str(function_obj.get("name") or "").strip() or None
        if raw_arguments is None:
            raw_arguments = function_obj.get("arguments")
    elif isinstance(function_obj, str):
        function_name = function_obj.strip() or None
    if function_name is None:
        function_name = "bash"

    if isinstance(raw_arguments, str):
        arguments_str = raw_arguments
    elif isinstance(raw_arguments, dict):
        arguments_str = json.dumps(raw_arguments, ensure_ascii=False)
    elif raw_arguments is None:
        arguments_str = "{}"
    else:
        arguments_str = json.dumps(raw_arguments, ensure_ascii=False, default=str)

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": arguments_str,
        },
    }


def normalize_docent_message_for_model(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    normalized: dict[str, Any] = {
        "role": role,
        "content": message.get("content"),
    }
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            parsed_calls = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    parsed = _normalize_tool_call(tool_call)
                    if parsed is not None:
                        parsed_calls.append(parsed)
            if parsed_calls:
                normalized["tool_calls"] = parsed_calls
    elif role == "tool":
        tool_call_id = message.get("tool_call_id")
        if tool_call_id is not None:
            normalized["tool_call_id"] = tool_call_id
        name = message.get("name") or message.get("tool_name") or message.get("function")
        if name is not None:
            normalized["name"] = name
    return normalized


def normalize_docent_messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_docent_message_for_model(message) for message in messages if isinstance(message, dict)]


def _extract_command_from_tool_call(tool_call: dict[str, Any]) -> str | None:
    raw_arguments = tool_call.get("arguments")
    function_obj = tool_call.get("function")
    if isinstance(function_obj, dict):
        raw_arguments = function_obj.get("arguments", raw_arguments)

    if isinstance(raw_arguments, dict):
        command = raw_arguments.get("command")
        return command if isinstance(command, str) else None

    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            command = parsed.get("command")
            return command if isinstance(command, str) else None
    return None


def extract_actions_from_assistant_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return actions
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        command = _extract_command_from_tool_call(tool_call)
        if not command:
            continue
        action: dict[str, Any] = {"command": command}
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str):
            action["tool_call_id"] = tool_call_id
        actions.append(action)
    return actions


def extract_replay_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    step_index = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        actions = extract_actions_from_assistant_message(message)
        if not actions:
            continue
        steps.append(
            {
                "step_index": step_index,
                "message_index": message_index,
                "assistant_message": message,
                "gold_actions": actions,
            }
        )
        step_index += 1
    return steps


def trajectory_contains_parallel_tool_calls(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            return True
    return False


def _extract_usage(response_message: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] = {"api_calls": 1}
    extra = response_message.get("extra")
    if not isinstance(extra, dict):
        return usage

    if isinstance(extra.get("cost"), (int, float)):
        usage["cost"] = float(extra["cost"])
    response = extra.get("response")
    if not isinstance(response, dict):
        return usage
    raw_usage = response.get("usage")
    if isinstance(raw_usage, dict):
        usage["prompt_tokens"] = raw_usage.get("prompt_tokens")
        usage["completion_tokens"] = raw_usage.get("completion_tokens")
        usage["total_tokens"] = raw_usage.get("total_tokens")
    return usage


def _extract_sampled_actions(response_message: dict[str, Any]) -> list[dict[str, Any]]:
    extra = response_message.get("extra")
    if not isinstance(extra, dict):
        return []
    raw_actions = extra.get("actions")
    if not isinstance(raw_actions, list):
        return []
    actions: list[dict[str, Any]] = []
    for action in raw_actions:
        if not isinstance(action, dict):
            continue
        command = action.get("command")
        if not isinstance(command, str):
            continue
        parsed: dict[str, Any] = {"command": command}
        tool_call_id = action.get("tool_call_id")
        if isinstance(tool_call_id, str):
            parsed["tool_call_id"] = tool_call_id
        actions.append(parsed)
    return actions


def _json_dump_line(file_obj, row: dict[str, Any]) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False, default=str))
    file_obj.write("\n")


def _build_gold_candidate_row(
    *,
    instance_id: str,
    run_id: str | None,
    problem_id: str | None,
    trajectory_relpath: str,
    step_index: int,
    message_index: int,
    prompt_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    gold_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    action_text = "\n".join(action["command"] for action in gold_actions)
    return {
        "dataset_version": DATASET_VERSION,
        "instance_id": instance_id,
        "run_id": run_id,
        "problem_id": problem_id,
        "trajectory_relpath": trajectory_relpath,
        "step_index": step_index,
        "message_index": message_index,
        "candidate_source": "gold",
        "is_gold": True,
        "sampler_model_id": "gold",
        "sampler_model_name": None,
        "sample_index": None,
        "prompt_messages": prompt_messages,
        "candidate_message": normalize_docent_message_for_model(assistant_message),
        "actions": gold_actions,
        "action_text": action_text,
        "has_actions": bool(gold_actions),
        "error": None,
        "usage": None,
        "created_at": int(time.time()),
    }


def _build_sampled_candidate_row(
    *,
    instance_id: str,
    run_id: str | None,
    problem_id: str | None,
    trajectory_relpath: str,
    step_index: int,
    message_index: int,
    prompt_messages: list[dict[str, Any]],
    sampler_spec: SamplerModelSpec,
    sample_index: int,
    response_message: dict[str, Any] | None,
    error_payload: dict[str, str] | None,
) -> dict[str, Any]:
    actions = _extract_sampled_actions(response_message or {})
    action_text = "\n".join(action["command"] for action in actions)
    return {
        "dataset_version": DATASET_VERSION,
        "instance_id": instance_id,
        "run_id": run_id,
        "problem_id": problem_id,
        "trajectory_relpath": trajectory_relpath,
        "step_index": step_index,
        "message_index": message_index,
        "candidate_source": "sampled",
        "is_gold": False,
        "sampler_model_id": sampler_spec.id,
        "sampler_model_name": sampler_spec.model_name,
        "sample_index": sample_index,
        "prompt_messages": prompt_messages,
        "candidate_message": response_message,
        "actions": actions,
        "action_text": action_text,
        "has_actions": bool(actions),
        "error": error_payload,
        "usage": _extract_usage(response_message or {}) if response_message else None,
        "created_at": int(time.time()),
    }


def _sample_candidate(
    *,
    model: Any,
    sampler_spec: SamplerModelSpec,
    prompt_messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        response_message = model.query(copy.deepcopy(prompt_messages), **sampler_spec.sampling_kwargs)
        if not isinstance(response_message, dict):
            raise TypeError(f"Model query returned non-dict response: {type(response_message).__name__}")
        return response_message, None
    except Exception as exc:  # noqa: BLE001
        traceback_text = traceback.format_exc(limit=10)
        return None, {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text[-4000:],
        }


def _prepare_runs(
    *,
    successful_runs: list[dict[str, Any]],
    transcripts_dir: Path,
    limit_steps_per_run: int | None,
    exclude_parallel_tool_call_trajectories: bool,
    counts: dict[str, int],
) -> list[PreparedRun]:
    prepared_runs: list[PreparedRun] = []
    for run_entry in successful_runs:
        try:
            if not isinstance(run_entry, dict):
                counts["runs_skipped"] += 1
                continue
            metadata = run_entry.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            transcript_refs = run_entry.get("transcripts")
            if not isinstance(transcript_refs, list) or not transcript_refs:
                counts["runs_skipped"] += 1
                continue
            transcript_ref = transcript_refs[0]
            if not isinstance(transcript_ref, str):
                counts["runs_skipped"] += 1
                continue

            transcript_path = _resolve_transcript_path(transcripts_dir, transcript_ref)
            transcript_obj = json.loads(transcript_path.read_text())
            transcript_obj_dict = transcript_obj if isinstance(transcript_obj, dict) else {}
            transcript = transcript_obj.get("transcript", transcript_obj) if isinstance(transcript_obj, dict) else {}
            messages = transcript.get("messages") if isinstance(transcript, dict) else None
            if not isinstance(messages, list):
                counts["runs_skipped"] += 1
                continue
            if exclude_parallel_tool_call_trajectories and trajectory_contains_parallel_tool_calls(messages):
                counts["runs_skipped_parallel_tool_calls"] += 1
                continue

            replay_steps = extract_replay_steps(messages)
            if limit_steps_per_run is not None:
                replay_steps = replay_steps[: max(0, limit_steps_per_run)]

            instance_id = str(metadata.get("instance_id") or transcript_obj_dict.get("problem_id") or "")
            if not instance_id:
                instance_id = transcript_path.stem
            run_id_value = run_entry.get("id")
            run_id = str(run_id_value) if run_id_value is not None else None
            problem_id_value = transcript_obj_dict.get("problem_id")
            problem_id = str(problem_id_value) if problem_id_value is not None else None
            try:
                trajectory_relpath = str(transcript_path.relative_to(transcripts_dir))
            except ValueError:
                trajectory_relpath = str(transcript_path)

            prepared_runs.append(
                PreparedRun(
                    instance_id=instance_id,
                    run_id=run_id,
                    problem_id=problem_id,
                    trajectory_relpath=trajectory_relpath,
                    messages=messages,
                    replay_steps=replay_steps,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("Skipping run due to unexpected error during pre-scan.")
            counts["runs_skipped"] += 1
    return prepared_runs


def generate_verifier_sampling_dataset(
    *,
    output_json_path: Path,
    transcripts_dir: Path,
    sampler_config_path: Path,
    output_dir: Path,
    num_samples: int = 2,
    max_workers: int = 8,
    limit_runs: int | None = None,
    limit_steps_per_run: int | None = None,
    exclude_parallel_tool_call_trajectories: bool = True,
    show_progress: bool = True,
    print_fct: Callable[[str], None] | None = print,
    overwrite: bool = False,
) -> dict[str, Any]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "candidates.jsonl"
    summary_json = output_dir / "summary.json"
    if not overwrite and output_jsonl.exists():
        raise FileExistsError(f"Output file already exists: {output_jsonl}")
    if not overwrite and summary_json.exists():
        raise FileExistsError(f"Summary file already exists: {summary_json}")

    sampler_specs = load_sampler_model_specs(sampler_config_path)
    model_instances = {
        spec.id: get_model(spec.model_name, config=copy.deepcopy(spec.model_config)) for spec in sampler_specs
    }

    output_payload = json.loads(output_json_path.read_text())
    if not isinstance(output_payload, list):
        raise ValueError("output.json must be a JSON list of run entries.")
    successful_runs = [entry for entry in output_payload if _is_resolved_run(entry)]
    if limit_runs is not None:
        successful_runs = successful_runs[: max(0, limit_runs)]

    counts: dict[str, int] = {
        "successful_runs_found": len([entry for entry in output_payload if _is_resolved_run(entry)]),
        "runs_processed": 0,
        "runs_skipped": 0,
        "runs_skipped_parallel_tool_calls": 0,
        "planned_runs": 0,
        "planned_steps": 0,
        "planned_sample_calls": 0,
        "steps_processed": 0,
        "gold_candidates": 0,
        "sample_candidates": 0,
        "sample_failures": 0,
    }
    failures_by_model: dict[str, int] = {spec.id: 0 for spec in sampler_specs}
    failures_by_type: dict[str, int] = {}

    prepared_runs = _prepare_runs(
        successful_runs=successful_runs,
        transcripts_dir=transcripts_dir,
        limit_steps_per_run=limit_steps_per_run,
        exclude_parallel_tool_call_trajectories=exclude_parallel_tool_call_trajectories,
        counts=counts,
    )
    counts["planned_runs"] = len(prepared_runs)
    counts["planned_steps"] = sum(len(run.replay_steps) for run in prepared_runs)
    counts["planned_sample_calls"] = counts["planned_steps"] * len(sampler_specs) * num_samples
    if print_fct is not None:
        print_fct(
            "Sampling plan: "
            f"runs={counts['planned_runs']} "
            f"steps={counts['planned_steps']} "
            f"models={len(sampler_specs)} "
            f"num_samples={num_samples} "
            f"total_model_calls={counts['planned_sample_calls']}"
        )

    run_progress = None
    step_progress = None
    if show_progress and _tqdm is not None:
        run_progress = _tqdm(total=counts["planned_runs"], desc="Runs", unit="run", leave=True)
        step_progress = _tqdm(total=counts["planned_steps"], desc="Steps", unit="step", leave=True)

    started_at = time.time()
    with output_jsonl.open("w", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for prepared_run in prepared_runs:
                try:
                    for step in prepared_run.replay_steps:
                        step_index = int(step["step_index"])
                        message_index = int(step["message_index"])
                        assistant_message = step["assistant_message"]
                        gold_actions = step["gold_actions"]
                        prompt_messages = normalize_docent_messages_for_model(prepared_run.messages[:message_index])

                        gold_row = _build_gold_candidate_row(
                            instance_id=prepared_run.instance_id,
                            run_id=prepared_run.run_id,
                            problem_id=prepared_run.problem_id,
                            trajectory_relpath=prepared_run.trajectory_relpath,
                            step_index=step_index,
                            message_index=message_index,
                            prompt_messages=prompt_messages,
                            assistant_message=assistant_message,
                            gold_actions=gold_actions,
                        )
                        _json_dump_line(output_file, gold_row)
                        counts["gold_candidates"] += 1
                        counts["steps_processed"] += 1

                        futures = {}
                        for sampler_spec in sampler_specs:
                            model = model_instances[sampler_spec.id]
                            for sample_index in range(num_samples):
                                future = executor.submit(
                                    _sample_candidate,
                                    model=model,
                                    sampler_spec=sampler_spec,
                                    prompt_messages=prompt_messages,
                                )
                                futures[future] = (sampler_spec, sample_index)

                        for future in as_completed(futures):
                            sampler_spec, sample_index = futures[future]
                            response_message, error_payload = future.result()
                            row = _build_sampled_candidate_row(
                                instance_id=prepared_run.instance_id,
                                run_id=prepared_run.run_id,
                                problem_id=prepared_run.problem_id,
                                trajectory_relpath=prepared_run.trajectory_relpath,
                                step_index=step_index,
                                message_index=message_index,
                                prompt_messages=prompt_messages,
                                sampler_spec=sampler_spec,
                                sample_index=sample_index,
                                response_message=response_message,
                                error_payload=error_payload,
                            )
                            _json_dump_line(output_file, row)
                            counts["sample_candidates"] += 1
                            if error_payload is not None:
                                counts["sample_failures"] += 1
                                failures_by_model[sampler_spec.id] = failures_by_model.get(sampler_spec.id, 0) + 1
                                error_type = error_payload.get("type", "UnknownError")
                                failures_by_type[error_type] = failures_by_type.get(error_type, 0) + 1
                        if step_progress is not None:
                            step_progress.update(1)
                    counts["runs_processed"] += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Skipping run due to unexpected error during sampling.")
                    counts["runs_skipped"] += 1
                finally:
                    if run_progress is not None:
                        run_progress.update(1)

    if run_progress is not None:
        run_progress.close()
    if step_progress is not None:
        step_progress.close()

    duration_seconds = time.time() - started_at
    summary = {
        "dataset_version": DATASET_VERSION,
        "input": {
            "output_json_path": str(output_json_path),
            "transcripts_dir": str(transcripts_dir),
            "sampler_config_path": str(sampler_config_path),
        },
        "config": {
            "num_samples": num_samples,
            "max_workers": max_workers,
            "limit_runs": limit_runs,
            "limit_steps_per_run": limit_steps_per_run,
            "exclude_parallel_tool_call_trajectories": exclude_parallel_tool_call_trajectories,
            "show_progress": show_progress,
            "sampler_models": [
                {
                    "id": spec.id,
                    "model_name": spec.model_name,
                    "model_config": spec.model_config,
                    "sampling_kwargs": spec.sampling_kwargs,
                }
                for spec in sampler_specs
            ],
        },
        "counts": {
            **counts,
            "total_candidates": counts["gold_candidates"] + counts["sample_candidates"],
        },
        "failures_by_model": failures_by_model,
        "failures_by_type": failures_by_type,
        "duration_seconds": duration_seconds,
        "output_jsonl": str(output_jsonl),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

from minisweagent.agents.default import VerifierConfig
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.utils.langfuse import attach_langfuse_session_metadata, enable_langfuse_tracing, make_langfuse_session_id
from minisweagent.utils.serialize import UNSET, recursive_merge
from minisweagent.verifiers.checklist import generate_issue_checklist

app = typer.Typer(add_completion=False)


def _default_prompt_name(generator_mode: str) -> str:
    return {
        "trajectory_success": "static_success",
        "trajectory_failure": "static_failure",
        "trajectory_pairwise": "pairwise_evolve",
        "trajectory_dynamic": "dynamic_success",
    }.get(generator_mode, "static_success")


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix == ".jsonl":
        return [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    data = json.loads(input_path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Checklist input must be a JSON object, array, or JSONL file.")


def _extract_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [message for message in payload if isinstance(message, dict)]
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    raise ValueError("Trajectory payload must be a list of messages or an object with a 'messages' list.")


def _extract_task_from_user_message(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""

    pr_description_match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", text, re.DOTALL | re.IGNORECASE)
    extracted = pr_description_match.group(1).strip() if pr_description_match is not None else text
    extracted = re.sub(
        r"^\s*Consider the following PR description:\s*",
        "",
        extracted,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return extracted or text


def _extract_task(payload: Any, messages: list[dict[str, Any]]) -> str:
    if isinstance(payload, dict):
        for key in ("task", "problem_statement"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _extract_task_from_user_message(value)
        info = payload.get("info", {})
        if isinstance(info, dict):
            value = info.get("task")
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


def _flatten_steps(steps: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for step in steps for message in step]


def _load_trajectory_context(row: dict[str, Any]) -> dict[str, Any]:
    trajectory_path = row.get("trajectory_path")
    if not isinstance(trajectory_path, str) or not trajectory_path.strip():
        return row
    traj_data = json.loads(Path(trajectory_path).read_text())
    messages = _extract_messages(traj_data)
    steps = _messages_to_steps(messages)
    merged = dict(row)
    merged.setdefault("task", _extract_task(traj_data, messages))
    merged.setdefault("messages", messages)
    merged.setdefault("all_messages", messages)
    merged.setdefault("steps", steps)
    merged.setdefault("all_steps", steps)
    return merged


def _build_config(
    *,
    config_spec: list[str],
    model_name: str | None,
    model_class: str | None,
    generator_mode: str,
    prompt_name: str,
) -> VerifierConfig:
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {
                "verifier": {
                    "model": {
                        "model_name": model_name or UNSET,
                        "model_class": model_class or UNSET,
                    },
                    "checklist_mode": "issue_progress",
                    "checklist_generator_mode": generator_mode,
                    "checklist_generator_prompt_name": prompt_name,
                    "checklist_output_format": "rubric_yaml",
                    "include_inputs_in_output": True,
                }
            }
        }
    )
    merged = recursive_merge(*configs)
    verifier_config = merged.get("agent", {}).get("verifier", {})
    return VerifierConfig(**verifier_config)


@app.command()
def main(
    input_path: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        dir_okay=False,
        help="JSON or JSONL checklist inputs.",
    ),
    trajectory: Path | None = typer.Option(
        None,
        "--trajectory",
        exists=True,
        dir_okay=False,
        help="Single trajectory JSON file to turn into a checklist artifact.",
    ),
    output_path: Path = typer.Option(..., "--output", dir_okay=False, help="Output JSONL path"),
    generator_mode: str = typer.Option(
        "trajectory_success",
        "--mode",
        help="One of trajectory_success, trajectory_failure, trajectory_pairwise, trajectory_dynamic",
    ),
    prompt_name: str = typer.Option("", "--prompt-name", help="Override checklist generator prompt family name"),
    compare_trajectory: Path | None = typer.Option(
        None,
        "--compare-trajectory",
        exists=True,
        dir_okay=False,
        help="Comparison trajectory for trajectory_pairwise mode.",
    ),
    existing_rubric: Path | None = typer.Option(
        None,
        "--existing-rubric",
        exists=True,
        dir_okay=False,
        help="Optional existing rubric text/YAML file.",
    ),
    step_index: int | None = typer.Option(
        None,
        "--step-index",
        help="For trajectory_dynamic mode: number of completed steps to treat as prior context.",
    ),
    model_name: str | None = typer.Option(None, "--model", help="Verifier model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Verifier model class"),
    enable_langfuse: bool = typer.Option(
        False,
        "--enable-langfuse",
        help='Enable LiteLLM Langfuse tracing by adding "langfuse_otel" to litellm.callbacks',
    ),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="Config files or overrides to merge"),
) -> None:
    if input_path is None and trajectory is None:
        raise ValueError("Provide either --input or --trajectory.")
    if input_path is not None and trajectory is not None:
        raise ValueError("Use only one of --input or --trajectory.")

    prompt_name = prompt_name or _default_prompt_name(generator_mode)
    verifier_config = _build_config(
        config_spec=config_spec,
        model_name=model_name,
        model_class=model_class,
        generator_mode=generator_mode,
        prompt_name=prompt_name,
    )
    if enable_langfuse:
        enable_langfuse_tracing()
        session_id = make_langfuse_session_id(
            prefix="trajectory-checklists",
            output_path=output_path,
            parts=[generator_mode, prompt_name],
        )
        attach_langfuse_session_metadata({"model": verifier_config.model}, session_id=session_id)
    model = get_model(config=dict(verifier_config.model))
    rows = [_load_trajectory_context(row) for row in _load_rows(input_path)] if input_path is not None else []
    if trajectory is not None:
        payload = json.loads(trajectory.read_text())
        messages = _extract_messages(payload)
        steps = _messages_to_steps(messages)
        row = {
            "task": _extract_task(payload, messages),
            "messages": messages,
            "all_messages": messages,
            "steps": steps,
            "all_steps": steps,
            "trajectory_path": str(trajectory),
        }
        if compare_trajectory is not None:
            compare_payload = json.loads(compare_trajectory.read_text())
            compare_messages = _extract_messages(compare_payload)
            compare_steps = _messages_to_steps(compare_messages)
            row["compare_trajectory_text"] = "\n".join(
                f"{message.get('role', 'unknown')}: {message.get('content', '')}"
                for message in _flatten_steps(compare_steps)
            )
        if existing_rubric is not None:
            row["existing_rubric"] = existing_rubric.read_text()
        if generator_mode == "trajectory_dynamic":
            if step_index is None:
                raise ValueError("--step-index is required for trajectory_dynamic mode.")
            if step_index < 0 or step_index > len(steps):
                raise ValueError(f"--step-index must be between 0 and {len(steps)}.")
            row["steps"] = steps[:step_index]
            row["messages"] = _flatten_steps(row["steps"])
        rows = [row]

    outputs: list[str] = []
    for row in rows:
        result = generate_issue_checklist(
            model,
            verifier_config,
            template_vars=row,
        )
        outputs.append(json.dumps({"input": row, "output": result, "prompt_name": prompt_name}))
    output_path.write_text("\n".join(outputs) + ("\n" if outputs else ""))


if __name__ == "__main__":
    app()

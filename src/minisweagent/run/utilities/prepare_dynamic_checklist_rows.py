from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT_DIR = Path("/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/minimax_checklist")
DEFAULT_SUCCESS_INPUT = DEFAULT_OUTPUT_DIR / "success_rows.jsonl"
DEFAULT_FAILURE_INPUT = DEFAULT_OUTPUT_DIR / "static_failure_rows.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _flatten_steps(steps: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for step in steps for message in step]


def _summarize_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    content = " ".join(str(message.get("content") or "").split())
    return f"{role}: {content}" if content else role


def _render_future_steps(steps: list[list[dict[str, Any]]]) -> list[str]:
    return [_summarize_message(message) for step in steps for message in step]


def _expand_row(
    row: dict[str, Any],
    *,
    dynamic_source: str,
    dynamic_source_category: str,
) -> list[dict[str, Any]]:
    all_steps = row.get("all_steps")
    if not isinstance(all_steps, list):
        raise ValueError(f"Dynamic row expansion requires all_steps in row for {row.get('instance_id', '<unknown>')}")

    normalized_steps: list[list[dict[str, Any]]] = []
    for step in all_steps:
        if isinstance(step, list):
            normalized_steps.append([message for message in step if isinstance(message, dict)])
        else:
            normalized_steps.append([])

    outputs: list[dict[str, Any]] = []
    total_steps = len(normalized_steps)
    for step_index in range(total_steps + 1):
        metadata = dict(row.get("selection_metadata", {}))
        metadata["dynamic_step_index"] = step_index
        metadata["dynamic_total_steps"] = total_steps
        metadata["dynamic_source"] = dynamic_source
        metadata["dynamic_source_category"] = dynamic_source_category
        expanded = {
            "instance_id": row.get("instance_id"),
            "seed": row.get("seed"),
            "trajectory_path": row.get("trajectory_path"),
            "task": row.get("task"),
            "step_index": step_index,
            "selection_metadata": metadata,
        }
        outputs.append(expanded)
    return outputs


def prepare_dynamic_rows(
    *,
    success_input: Path,
    failure_input: Path,
    output_dir: Path,
) -> dict[str, int]:
    success_rows = _load_rows(success_input)
    failure_rows = _load_rows(failure_input)

    dynamic_success_rows: list[dict[str, Any]] = []
    for row in success_rows:
        dynamic_success_rows.extend(
            _expand_row(
                row,
                dynamic_source=success_input.name,
                dynamic_source_category="dynamic_success",
            )
        )

    dynamic_failure_rows: list[dict[str, Any]] = []
    for row in failure_rows:
        dynamic_failure_rows.extend(
            _expand_row(
                row,
                dynamic_source=failure_input.name,
                dynamic_source_category="dynamic_failure",
            )
        )

    _write_jsonl(output_dir / "dynamic_success_rows.jsonl", dynamic_success_rows)
    _write_jsonl(output_dir / "dynamic_failure_rows.jsonl", dynamic_failure_rows)
    return {
        "dynamic_success_rows": len(dynamic_success_rows),
        "dynamic_failure_rows": len(dynamic_failure_rows),
    }


@app.command()
def main(
    success_input: Path = typer.Option(DEFAULT_SUCCESS_INPUT, "--success-input", exists=True, dir_okay=False),
    failure_input: Path = typer.Option(DEFAULT_FAILURE_INPUT, "--failure-input", exists=True, dir_okay=False),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir", file_okay=False),
) -> None:
    counts = prepare_dynamic_rows(
        success_input=success_input,
        failure_input=failure_input,
        output_dir=output_dir,
    )
    typer.echo(
        "Wrote dynamic checklist rows to "
        f"{output_dir} (success={counts['dynamic_success_rows']}, failure={counts['dynamic_failure_rows']})"
    )


if __name__ == "__main__":
    app()

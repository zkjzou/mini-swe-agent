#!/usr/bin/env python3

"""Add average Monte Carlo rollout steps to each action entry in an action_summary JSON."""

from __future__ import annotations

import json
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _population_std(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _load_rollout_step_averages(results_jsonl: Path) -> dict[tuple[str, int, int], dict[str, float | int]]:
    steps_by_action: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            step_index = _safe_int(row.get("step_index"))
            action_index = _safe_int(row.get("action_index"))
            rollout_steps = _safe_int(row.get("rollout_executed_steps"))
            if not isinstance(instance_id, str) or step_index is None or action_index is None or rollout_steps is None:
                continue
            steps_by_action[(instance_id, step_index, action_index)].append(rollout_steps)

    averages: dict[tuple[str, int, int], dict[str, float | int]] = {}
    for key, values in steps_by_action.items():
        averages[key] = {
            "rollout_executed_steps_values": values,
            "avg_rollout_executed_steps": sum(values) / len(values),
            "rollout_executed_steps_std": _population_std(values),
            "rollout_samples": len(values),
        }
    return averages


def add_rollout_steps_to_action_summary(
    action_summary: dict[str, Any],
    *,
    results_jsonl: Path,
) -> dict[str, Any]:
    rollout_step_averages = _load_rollout_step_averages(results_jsonl)
    updated_summary = json.loads(json.dumps(action_summary))

    for instance_id, instance_summary in updated_summary.items():
        if not isinstance(instance_summary, dict):
            continue
        for step_key, step_summary in instance_summary.items():
            if not isinstance(step_summary, dict) or not step_key.startswith("step_"):
                continue
            step_index = _safe_int(step_key.removeprefix("step_"))
            if step_index is None:
                continue
            for action_key, action_summary_entry in step_summary.items():
                if not isinstance(action_summary_entry, dict) or not action_key.startswith("action_"):
                    continue
                action_index = _safe_int(action_key.removeprefix("action_"))
                if action_index is None:
                    continue
                rollout_stats = rollout_step_averages.get((instance_id, step_index, action_index))
                if rollout_stats is None:
                    continue
                action_summary_entry.update(rollout_stats)

    return updated_summary


@app.command(help=__doc__)
def main(
    input_json: str = typer.Option(..., "--input-json", help="Input action_summary.json path"),
    results_jsonl: str = typer.Option(..., "--results-jsonl", help="Monte Carlo rollout results.jsonl path"),
    output_json: str = typer.Option(..., "--output-json", help="Output path for updated JSON"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the output file if it already exists"),
) -> None:
    input_path = Path(input_json)
    results_path = Path(results_jsonl)
    output_path = Path(output_json)
    if not input_path.is_file():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        raise typer.Exit(code=1)
    if not results_path.is_file():
        console.print(f"[red]Results JSONL not found:[/red] {results_path}")
        raise typer.Exit(code=1)
    if output_path.exists() and not overwrite:
        console.print(f"[red]Output file already exists:[/red] {output_path}")
        raise typer.Exit(code=1)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        console.print("[red]Input JSON must be a dictionary keyed by instance id.[/red]")
        raise typer.Exit(code=1)

    updated = add_rollout_steps_to_action_summary(payload, results_jsonl=results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote updated action summary:[/green] {output_path}")


if __name__ == "__main__":
    app()

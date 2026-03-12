#!/usr/bin/env python3

"""Identify critical steps from an action_summary-with-rollout-steps JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _load_action_metadata(results_jsonl: Path | None) -> dict[tuple[str, int, int], dict[str, str | None]]:
    metadata: dict[tuple[str, int, int], dict[str, str | None]] = {}
    if results_jsonl is None:
        return metadata
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            try:
                step_index = int(row.get("step_index"))
                action_index = int(row.get("action_index"))
            except (TypeError, ValueError):
                continue
            if not isinstance(instance_id, str):
                continue
            key = (instance_id, step_index, action_index)
            if key in metadata:
                continue
            forced_command = row.get("forced_command")
            forced_thought = row.get("forced_thought")
            metadata[key] = {
                "command": forced_command if isinstance(forced_command, str) else None,
                "thought": forced_thought if isinstance(forced_thought, str) else None,
            }
    return metadata


def _step_index_from_key(step_key: str) -> int | None:
    if not step_key.startswith("step_"):
        return None
    try:
        return int(step_key.removeprefix("step_"))
    except ValueError:
        return None


def _candidate_actions(
    instance_id: str,
    step_index: int,
    step_summary: dict[str, Any],
    *,
    action_metadata: dict[tuple[str, int, int], dict[str, str | None]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for action_key, action_data in sorted(step_summary.items()):
        if not action_key.startswith("action_") or not isinstance(action_data, dict):
            continue
        try:
            action_index = int(action_key.removeprefix("action_"))
        except ValueError:
            continue
        resolve_rate = _safe_float(action_data.get("resolve_rate"))
        avg_steps = _safe_float(action_data.get("avg_rollout_executed_steps"))
        if resolve_rate is None or avg_steps is None:
            continue
        extra_metadata = action_metadata.get((instance_id, step_index, action_index), {})
        actions.append(
            {
                "action_key": action_key,
                "label": action_data.get("label", action_key),
                "command": extra_metadata.get("command"),
                "thought": extra_metadata.get("thought"),
                "resolve_rate": resolve_rate,
                "resolved": action_data.get("resolved"),
                "unresolved": action_data.get("unresolved"),
                "error": action_data.get("error"),
                "total": action_data.get("total"),
                "avg_rollout_executed_steps": avg_steps,
                "rollout_executed_steps_std": _safe_float(action_data.get("rollout_executed_steps_std")) or 0.0,
                "rollout_samples": action_data.get("rollout_samples"),
                "rollout_executed_steps_values": action_data.get("rollout_executed_steps_values", []),
            }
        )
    return actions


def build_critical_step_report(
    payload: dict[str, Any],
    *,
    results_jsonl: Path | None = None,
    resolved_rate_range_threshold: float = 0.4,
    avg_rollout_steps_range_threshold: float = 40.0,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    skipped_steps_missing_metrics = 0
    instance_ids = [instance_id for instance_id, value in payload.items() if isinstance(value, dict)]
    action_metadata = _load_action_metadata(results_jsonl)

    for instance_id, instance_summary in payload.items():
        if not isinstance(instance_summary, dict):
            continue
        for step_key, step_summary in instance_summary.items():
            step_index = _step_index_from_key(step_key)
            if step_index is None or not isinstance(step_summary, dict):
                continue

            actions = _candidate_actions(
                instance_id,
                step_index,
                step_summary,
                action_metadata=action_metadata,
            )
            if len(actions) < 2:
                skipped_steps_missing_metrics += 1
                continue

            resolve_rates = [action["resolve_rate"] for action in actions]
            avg_steps = [action["avg_rollout_executed_steps"] for action in actions]
            resolved_rate_range = max(resolve_rates) - min(resolve_rates)
            avg_rollout_steps_range = max(avg_steps) - min(avg_steps)
            best_resolved_rate_label = max(actions, key=lambda action: (action["resolve_rate"], -action["avg_rollout_executed_steps"]))[
                "label"
            ]
            shortest_avg_rollout_steps_label = min(
                actions,
                key=lambda action: (action["avg_rollout_executed_steps"], -action["resolve_rate"]),
            )["label"]

            entry = {
                "instance_id": instance_id,
                "step_key": step_key,
                "step_index": step_index,
                "critical_point": (
                    resolved_rate_range >= resolved_rate_range_threshold
                    and avg_rollout_steps_range >= avg_rollout_steps_range_threshold
                ),
                "resolved_rate_min": min(resolve_rates),
                "resolved_rate_max": max(resolve_rates),
                "resolved_rate_range": resolved_rate_range,
                "resolved_rate_std": _population_std(resolve_rates),
                "avg_rollout_steps_min": min(avg_steps),
                "avg_rollout_steps_max": max(avg_steps),
                "avg_rollout_steps_range": avg_rollout_steps_range,
                "avg_rollout_steps_std": _population_std(avg_steps),
                "best_resolved_rate_label": best_resolved_rate_label,
                "shortest_avg_rollout_steps_label": shortest_avg_rollout_steps_label,
                "winner_disagreement": best_resolved_rate_label != shortest_avg_rollout_steps_label,
                "candidate_actions": actions,
            }
            entries.append(entry)

    entries.sort(
        key=lambda entry: (
            not entry["critical_point"],
            -entry["resolved_rate_range"],
            -entry["avg_rollout_steps_range"],
            entry["instance_id"],
            entry["step_index"],
        )
    )
    critical_steps = sum(1 for entry in entries if entry["critical_point"])
    return {
        "n_instances": len(instance_ids),
        "n_steps": len(entries),
        "n_critical_steps": critical_steps,
        "critical_step_fraction": (critical_steps / len(entries)) if entries else 0.0,
        "skipped_steps_missing_metrics": skipped_steps_missing_metrics,
        "resolved_rate_range_threshold": resolved_rate_range_threshold,
        "avg_rollout_steps_range_threshold": avg_rollout_steps_range_threshold,
        "steps": entries,
    }


@app.command(help=__doc__)
def main(
    input_json: str = typer.Option(..., "--input-json", help="Input action_summary.with_rollout_steps.json path"),
    results_jsonl: str | None = typer.Option(
        None,
        "--results-jsonl",
        help="Optional Monte Carlo results.jsonl for attaching candidate commands and thoughts",
    ),
    output_json: str = typer.Option(..., "--output-json", help="Output path for critical step report"),
    resolved_rate_range_threshold: float = typer.Option(
        0.4,
        "--resolved-rate-range-threshold",
        min=0.0,
        max=1.0,
        help="Minimum resolve-rate spread required for a critical step",
    ),
    avg_rollout_steps_range_threshold: float = typer.Option(
        40.0,
        "--avg-rollout-steps-range-threshold",
        min=0.0,
        help="Minimum average rollout-step spread required for a critical step",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the output file if it already exists"),
) -> None:
    input_path = Path(input_json)
    results_path = Path(results_jsonl) if results_jsonl else None
    output_path = Path(output_json)
    if not input_path.is_file():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        raise typer.Exit(code=1)
    if results_path is not None and not results_path.is_file():
        console.print(f"[red]Results JSONL not found:[/red] {results_path}")
        raise typer.Exit(code=1)
    if output_path.exists() and not overwrite:
        console.print(f"[red]Output file already exists:[/red] {output_path}")
        raise typer.Exit(code=1)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        console.print("[red]Input JSON must be a dictionary keyed by instance id.[/red]")
        raise typer.Exit(code=1)

    report = build_critical_step_report(
        payload,
        results_jsonl=results_path,
        resolved_rate_range_threshold=resolved_rate_range_threshold,
        avg_rollout_steps_range_threshold=avg_rollout_steps_range_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote critical step report:[/green] {output_path}")


if __name__ == "__main__":
    app()

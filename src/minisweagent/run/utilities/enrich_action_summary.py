#!/usr/bin/env python3

"""Enrich an action_summary JSON with step-level/action-level variability and critical-point metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

_OUTCOME_KEYS = ("resolved", "unresolved", "error")


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _outcome_rates(action_summary: dict[str, Any]) -> dict[str, float]:
    total = max(1, _safe_int(action_summary.get("total")))
    return {key: _safe_int(action_summary.get(key)) / total for key in _OUTCOME_KEYS}


def _max_total_variation(actions: list[tuple[str, dict[str, Any]]]) -> tuple[float, tuple[str, str] | None]:
    max_distance = 0.0
    max_pair: tuple[str, str] | None = None
    for left_index, (left_key, left_action) in enumerate(actions):
        left_rates = _outcome_rates(left_action)
        for right_key, right_action in actions[left_index + 1 :]:
            right_rates = _outcome_rates(right_action)
            distance = 0.5 * sum(abs(left_rates[key] - right_rates[key]) for key in _OUTCOME_KEYS)
            if distance > max_distance:
                max_distance = distance
                max_pair = (left_key, right_key)
    return max_distance, max_pair


def _enrich_action(action_key: str, action_summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(action_summary)
    total = max(0, _safe_int(action_summary.get("total")))
    resolve_rate = _safe_float(action_summary.get("resolve_rate"))
    if total > 0 and "resolve_rate" not in action_summary:
        resolve_rate = _safe_int(action_summary.get("resolved")) / total
        enriched["resolve_rate"] = resolve_rate
    enriched["n_rollouts"] = total
    enriched["resolve_rate_std"] = math.sqrt(resolve_rate * (1.0 - resolve_rate)) if total > 0 else 0.0
    enriched["outcome_rates"] = _outcome_rates(action_summary)
    enriched["action_key"] = action_key
    return enriched


def _critical_summary(
    actions: list[tuple[str, dict[str, Any]]],
    *,
    critical_threshold: float,
) -> dict[str, Any]:
    resolve_rates = [_safe_float(action.get("resolve_rate")) for _, action in actions]
    unique_profiles = {
        tuple(_safe_int(action.get(key)) for key in _OUTCOME_KEYS)
        for _, action in actions
    }
    max_tv_distance, max_pair = _max_total_variation(actions)
    is_critical = len(unique_profiles) > 1 and max_tv_distance >= critical_threshold
    return {
        "n_actions": len(actions),
        "resolve_rate_mean": (sum(resolve_rates) / len(resolve_rates)) if resolve_rates else 0.0,
        "resolve_rate_std": _population_std(resolve_rates),
        "resolve_rate_min": min(resolve_rates) if resolve_rates else 0.0,
        "resolve_rate_max": max(resolve_rates) if resolve_rates else 0.0,
        "resolve_rate_range": (max(resolve_rates) - min(resolve_rates)) if resolve_rates else 0.0,
        "n_unique_outcome_profiles": len(unique_profiles),
        "max_outcome_total_variation": max_tv_distance,
        "critical_threshold": critical_threshold,
        "critical_point": is_critical,
        "critical_pair": list(max_pair) if max_pair is not None else None,
        "critical_reason": (
            "actions have materially different outcome distributions"
            if is_critical
            else "action outcomes are too similar under the configured threshold"
        ),
    }


def enrich_action_summary(payload: dict[str, Any], *, critical_threshold: float = 0.4) -> dict[str, Any]:
    enriched_payload: dict[str, Any] = {}
    for instance_id, instance_summary in payload.items():
        if not isinstance(instance_summary, dict):
            enriched_payload[instance_id] = instance_summary
            continue

        enriched_instance: dict[str, Any] = {}
        step_keys = sorted(key for key in instance_summary if not key.startswith("__"))
        critical_steps = 0
        step_std_values: list[float] = []

        for step_key in step_keys:
            step_summary = instance_summary.get(step_key)
            if not isinstance(step_summary, dict):
                enriched_instance[step_key] = step_summary
                continue

            action_items = [
                (action_key, action_value)
                for action_key, action_value in sorted(step_summary.items())
                if isinstance(action_value, dict) and not action_key.startswith("__")
            ]
            enriched_step: dict[str, Any] = {}
            enriched_actions = [(action_key, _enrich_action(action_key, action_value)) for action_key, action_value in action_items]
            for action_key, enriched_action in enriched_actions:
                enriched_step[action_key] = enriched_action

            step_meta = _critical_summary(enriched_actions, critical_threshold=critical_threshold)
            enriched_step["__step_summary__"] = step_meta
            enriched_instance[step_key] = enriched_step
            critical_steps += int(step_meta["critical_point"])
            step_std_values.append(_safe_float(step_meta["resolve_rate_std"]))

        enriched_instance["__instance_summary__"] = {
            "n_steps": len(step_keys),
            "n_critical_steps": critical_steps,
            "critical_step_fraction": (critical_steps / len(step_keys)) if step_keys else 0.0,
            "step_resolve_rate_std_mean": (sum(step_std_values) / len(step_std_values)) if step_std_values else 0.0,
            "step_resolve_rate_std_max": max(step_std_values) if step_std_values else 0.0,
            "critical_threshold": critical_threshold,
        }
        enriched_payload[instance_id] = enriched_instance

    return enriched_payload


@app.command(help=__doc__)
def main(
    input_json: str = typer.Option(..., "--input-json", help="Input action_summary.json path"),
    output_json: str = typer.Option(..., "--output-json", help="Output path for enriched JSON"),
    critical_threshold: float = typer.Option(
        0.4,
        "--critical-threshold",
        min=0.0,
        max=1.0,
        help="Minimum total-variation divergence needed to mark a step as critical",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the output file if it already exists"),
) -> None:
    input_path = Path(input_json)
    output_path = Path(output_json)
    if not input_path.is_file():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        raise typer.Exit(code=1)
    if output_path.exists() and not overwrite:
        console.print(f"[red]Output file already exists:[/red] {output_path}")
        raise typer.Exit(code=1)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        console.print("[red]Input JSON must be a dictionary keyed by instance id.[/red]")
        raise typer.Exit(code=1)

    enriched = enrich_action_summary(payload, critical_threshold=critical_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote enriched action summary:[/green] {output_path}")


if __name__ == "__main__":
    app()

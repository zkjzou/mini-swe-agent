#!/usr/bin/env python3

"""Reanalyze verifier action distributions using rollout-derived action rankings."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

_RANK_NAMES = ("gold", "2nd", "3rd", "4th", "5th")
_DISPLAY_RANK_NAMES = {rank: rank for rank in _RANK_NAMES}
_AGGREGATED_SCORE_WEIGHTS = {
    "gold": 1.0,
    "2nd": 0.75,
    "3rd": 0.5,
    "4th": 0.25,
    "5th": 0.0,
}


def _display_rank_name(rank: str) -> str:
    return _DISPLAY_RANK_NAMES.get(rank, rank)


def _display_model_name(output_jsonl: Path, verifier_type: str) -> str:
    if output_jsonl.is_relative_to(Path("/tmp")):
        return verifier_type
    raw_name = output_jsonl.parent.name
    if raw_name == "verifier_samples":
        raw_name = output_jsonl.stem.removesuffix("_rows")
        if raw_name.startswith("verifier_eval_"):
            raw_name = raw_name.removeprefix("verifier_eval_")
    if not raw_name or raw_name.startswith("tmp"):
        return verifier_type
    display_name = raw_name
    for source, target in (
        ("qwen3_5_27b", "qwen3.5-27b"),
        ("qwen3_5_35b", "qwen3.5-35b"),
        ("qwen3_5_instruct", "qwen3.5-27b"),
        ("gpt5_mini", "gpt5-mini"),
        ("gpt5mini", "gpt5-mini"),
        ("qwen3_coder_next", "qwen3-coder-next"),
        ("qwen3_coder", "qwen3-coder-instruct"),
    ):
        display_name = display_name.replace(source, target)
    return display_name.replace("_", "-")


def _selected_label(row: dict) -> str:
    selected_label = row.get("selected_label")
    if isinstance(selected_label, str) and selected_label:
        return selected_label
    return f"index_{row.get('selected_index')}"


def _n_candidates(row: dict) -> int | None:
    n_actions = row.get("n_actions")
    if isinstance(n_actions, int) and n_actions > 0:
        return n_actions
    candidate_labels = row.get("candidate_labels")
    if isinstance(candidate_labels, list) and candidate_labels:
        return len(candidate_labels)
    return None


def _has_parser_failure(row: dict) -> bool:
    if row.get("verifier_type") != "llm":
        return False
    verifier_output = row.get("verifier_output")
    if not isinstance(verifier_output, dict):
        return False
    raw_index = verifier_output.get("raw_index")
    if raw_index is None:
        return True
    try:
        parsed_raw_index = int(raw_index)
    except (TypeError, ValueError):
        return True
    n_candidates = _n_candidates(row)
    if n_candidates is not None and not (1 <= parsed_raw_index <= n_candidates):
        return True
    return False


def _safe_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _step_index_from_key(step_key: str) -> int:
    prefix = "step_"
    if not step_key.startswith(prefix):
        raise ValueError(f"Unexpected step key format: {step_key}")
    return int(step_key[len(prefix) :])


def _action_sort_key(action_key: str, action_data: dict) -> tuple[float, float, str, str]:
    resolve_rate = _safe_float(action_data.get("resolve_rate"), default=0.0)
    avg_steps = _safe_float(action_data.get("avg_rollout_executed_steps"), default=inf)
    if not isfinite(avg_steps):
        avg_steps = inf
    label = str(action_data.get("label") or "")
    return (-resolve_rate, avg_steps, action_key, label)


def load_action_rankings(action_summary_json: Path) -> dict[tuple[str, int], dict[str, str]]:
    payload = json.loads(action_summary_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {action_summary_json}")

    rankings: dict[tuple[str, int], dict[str, str]] = {}
    for instance_id, steps_obj in payload.items():
        if not isinstance(instance_id, str) or not isinstance(steps_obj, dict):
            continue
        for step_key, actions_obj in steps_obj.items():
            if not isinstance(step_key, str) or not isinstance(actions_obj, dict):
                continue
            ranked_actions: list[tuple[str, dict, str]] = []
            for action_key, action_data in actions_obj.items():
                if not isinstance(action_key, str) or not isinstance(action_data, dict):
                    continue
                label = action_data.get("label")
                if not isinstance(label, str) or not label:
                    continue
                ranked_actions.append((action_key, action_data, label))

            ranked_actions.sort(key=lambda item: _action_sort_key(item[0], item[1]))
            if len(ranked_actions) < len(_RANK_NAMES):
                continue

            label_to_rank: dict[str, str] = {}
            for index, (_, _, label) in enumerate(ranked_actions[: len(_RANK_NAMES)]):
                label_to_rank[label] = _RANK_NAMES[index]
            rankings[(instance_id, _step_index_from_key(step_key))] = label_to_rank
    return rankings


def _make_distribution_column_names() -> dict[str, tuple[str, str]]:
    return {rank: (f"count__{_display_rank_name(rank)}", _display_rank_name(rank)) for rank in _RANK_NAMES}


def _compute_aggregated_score(parsed_counts: Counter[str], *, rows_evaluated: int) -> float:
    if rows_evaluated <= 0:
        return 0.0
    weighted_total = 0.0
    for rank, weight in _AGGREGATED_SCORE_WEIGHTS.items():
        weighted_total += parsed_counts.get(rank, 0) * weight
    return weighted_total / rows_evaluated


def collect_rank_distribution_rows(
    *, action_rankings: dict[tuple[str, int], dict[str, str]], output_jsonls: list[Path]
) -> list[dict[str, str]]:
    parsed_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    parser_failure_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str, str]] = Counter()
    ranked_totals: Counter[tuple[str, str, str]] = Counter()
    parsed_totals: Counter[tuple[str, str, str]] = Counter()
    parser_failures: Counter[tuple[str, str, str]] = Counter()
    missing_rankings: Counter[tuple[str, str, str]] = Counter()

    for output_jsonl in output_jsonls:
        with output_jsonl.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "evaluated":
                    continue

                verifier_type = str(row.get("verifier_type") or "")
                verifier_variant = str(row.get("verifier_variant") or verifier_type)
                key = (verifier_type, verifier_variant, str(output_jsonl))
                totals[key] += 1

                instance_id = row.get("instance_id")
                step_index = row.get("step_index")
                if not isinstance(instance_id, str) or not isinstance(step_index, int):
                    missing_rankings[key] += 1
                    continue

                rank_by_label = action_rankings.get((instance_id, step_index))
                if not rank_by_label:
                    missing_rankings[key] += 1
                    continue

                selected_rank = rank_by_label.get(_selected_label(row))
                if not selected_rank:
                    missing_rankings[key] += 1
                    continue

                ranked_totals[key] += 1
                if _has_parser_failure(row):
                    parser_failures[key] += 1
                    parser_failure_counts[key][selected_rank] += 1
                else:
                    parsed_totals[key] += 1
                    parsed_counts[key][selected_rank] += 1

    label_columns = _make_distribution_column_names()
    front_fieldnames = ["model", "verifier_variant"]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fraction_fieldnames = [_display_rank_name(rank) for rank in _RANK_NAMES]
    metadata_fieldnames = [
        "rows_evaluated",
        "rows_with_action_ranking",
        "rows_missing_action_ranking",
        "fraction_action_ranking",
        "rows_non_parser_failed",
        "rows_parser_failed",
        "fraction_parser_failed",
        "aggregated_score",
        "strict_gold_score",
        "timestamp_utc",
        "output_jsonl",
    ]
    count_fieldnames = [label_columns[rank][0] for rank in _RANK_NAMES]
    parser_failure_fraction_fieldnames = [
        f"parser_failure_fraction__{_display_rank_name(rank)}" for rank in _RANK_NAMES
    ]
    parser_failure_count_fieldnames = [f"parser_failure_count__{_display_rank_name(rank)}" for rank in _RANK_NAMES]
    fieldnames = (
        front_fieldnames
        + fraction_fieldnames
        + metadata_fieldnames
        + count_fieldnames
        + parser_failure_fraction_fieldnames
        + parser_failure_count_fieldnames
    )

    rows: list[dict[str, str]] = []
    for verifier_type, verifier_variant, output_jsonl in sorted(totals):
        key = (verifier_type, verifier_variant, output_jsonl)
        total = totals[key]
        ranked_total = ranked_totals[key]
        missing_total = missing_rankings[key]
        parsed_total = parsed_totals[key]
        parser_failed = parser_failures[key]
        row = {
            "model": _display_model_name(Path(output_jsonl), verifier_type),
            "verifier_variant": verifier_variant,
            "rows_evaluated": str(total),
            "rows_with_action_ranking": str(ranked_total),
            "rows_missing_action_ranking": str(missing_total),
            "fraction_action_ranking": f"{ranked_total / total:.6f}" if total else "0.000000",
            "rows_non_parser_failed": str(parsed_total),
            "rows_parser_failed": str(parser_failed),
            "fraction_parser_failed": f"{parser_failed / ranked_total:.6f}" if ranked_total else "0.000000",
            "aggregated_score": f"{_compute_aggregated_score(parsed_counts[key], rows_evaluated=total):.6f}",
            "strict_gold_score": f"{parsed_counts[key].get('gold', 0) / total:.6f}" if total else "0.000000",
            "timestamp_utc": timestamp,
            "output_jsonl": output_jsonl,
        }
        for rank in _RANK_NAMES:
            count_column, fraction_column = label_columns[rank]
            count = parsed_counts[key].get(rank, 0)
            row[count_column] = str(count)
            row[fraction_column] = f"{count / parsed_total:.6f}" if parsed_total else "0.000000"
        for rank in _RANK_NAMES:
            count = parser_failure_counts[key].get(rank, 0)
            rank_name = _display_rank_name(rank)
            row[f"parser_failure_count__{rank_name}"] = str(count)
            row[f"parser_failure_fraction__{rank_name}"] = (
                f"{count / parser_failed:.6f}" if parser_failed else "0.000000"
            )
        rows.append({field: row.get(field, "") for field in fieldnames})
    return rows


@app.command(help=__doc__)
def main(
    action_summary_json: str = typer.Option(..., "--action-summary-json", help="Action summary JSON with rollout steps"),
    input_jsonls: list[str] = typer.Option(
        None,
        "--input-jsonl",
        help="Verifier evaluation row JSONL file(s). Repeat for multiple files.",
    ),
    input_dir: str | None = typer.Option(
        None,
        "--input-dir",
        help="Directory containing *_rows.jsonl files. Used if --input-jsonl is omitted.",
    ),
    output_csv: str = typer.Option(..., "--output-csv", help="Output CSV path"),
) -> None:
    if input_jsonls:
        output_jsonls = [Path(path) for path in input_jsonls]
    elif input_dir:
        output_jsonls = sorted(Path(input_dir).glob("*_rows.jsonl"))
    else:
        raise typer.BadParameter("Provide either --input-jsonl or --input-dir")

    if not output_jsonls:
        raise typer.BadParameter("No input JSONL files found")

    rankings = load_action_rankings(Path(action_summary_json))
    rows = collect_rank_distribution_rows(action_rankings=rankings, output_jsonls=output_jsonls)
    if not rows:
        raise typer.BadParameter("No evaluated rows found in the provided JSONL files")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]Wrote rollout-ranked distribution CSV:[/green] {output_path}")
    for row in rows:
        console.print(
            "{model} {variant}: ranked={ranked}/{total} parser_failed={parser_failed} gold={gold}".format(
                model=row["model"],
                variant=row["verifier_variant"],
                ranked=row["rows_with_action_ranking"],
                total=row["rows_evaluated"],
                parser_failed=row["rows_parser_failed"],
                gold=row["gold"],
            )
        )


if __name__ == "__main__":
    app()

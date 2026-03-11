#!/usr/bin/env python3

"""Evaluate verifier gold-action selection accuracy from merged verifier-action rows."""

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.verifier_action_evaluation import evaluate_verifier_action_selection

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _sanitize_distribution_label(label: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(label or "").strip()).strip("_").lower()
    return sanitized or "unknown"


def _make_distribution_column_names(labels: list[str]) -> dict[str, tuple[str, str]]:
    columns: dict[str, tuple[str, str]] = {}
    seen: Counter[str] = Counter()
    for label in labels:
        base = _sanitize_distribution_label(label)
        seen[base] += 1
        suffix = f"_{seen[base]}" if seen[base] > 1 else ""
        key = f"{base}{suffix}"
        columns[label] = (f"count__{key}", f"fraction__{key}")
    return columns


def append_predicted_action_distribution(output_jsonl: Path, output_csv: Path) -> None:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
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
            selected_label = row.get("selected_label")
            if not isinstance(selected_label, str) or not selected_label:
                selected_label = f"index_{row.get('selected_index')}"
            key = (verifier_type, verifier_variant)
            counts[key][selected_label] += 1
            totals[key] += 1

    labels = sorted({label for label_counts in counts.values() for label in label_counts})
    label_columns = _make_distribution_column_names(labels)
    metadata_fieldnames = [
        "timestamp_utc",
        "verifier_type",
        "verifier_variant",
        "rows_evaluated",
        "output_jsonl",
    ]
    fieldnames = metadata_fieldnames + [
        column_name
        for label in labels
        for column_name in label_columns[label]
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing_rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = [dict(row) for row in reader]
    fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({field: existing_row.get(field, "") for field in fieldnames})
        for verifier_type, verifier_variant in sorted(counts):
            total = totals[(verifier_type, verifier_variant)]
            row = {
                "timestamp_utc": timestamp,
                "verifier_type": verifier_type,
                "verifier_variant": verifier_variant,
                "rows_evaluated": total,
                "output_jsonl": str(output_jsonl),
            }
            for selected_label, count in sorted(counts[(verifier_type, verifier_variant)].items()):
                count_column, fraction_column = label_columns[selected_label]
                row[count_column] = count
                row[fraction_column] = f"{count / total:.6f}" if total else "0.000000"
            writer.writerow({field: row.get(field, "") for field in fieldnames})


@app.command(help=__doc__)
def main(
    input_jsonl: str = typer.Option(..., "--input-jsonl", help="Merged verifier-action JSONL file"),
    output_jsonl: str = typer.Option(..., "--output-jsonl", help="Per-row evaluation JSONL output file"),
    output_summary: str | None = typer.Option(None, "--output-summary", help="Optional summary JSON output path"),
    output_distribution_csv: str | None = typer.Option(
        None,
        "--output-distribution-csv",
        help="Optional CSV file to append predicted action distributions from the evaluation output.",
    ),
    config_specs: list[str] = typer.Option(
        None,
        "-c",
        "--config",
        help=(
            "Config spec(s), e.g. swebench.yaml or key=value overrides. "
            "Repeat -c for multiple specs. Defaults to swebench benchmark config."
        ),
    ),
    verifier_types: list[str] = typer.Option(
        None,
        "--verifier-type",
        help="Verifier type(s) to evaluate. Repeat for multiple: first_valid, llm, reward_model",
    ),
    verifier_variants: list[str] = typer.Option(
        None,
        "--verifier-variant",
        help="Concrete verifier variant(s) to evaluate. Repeat for multiple; defaults to all current variants.",
    ),
    strict_five_actions: bool = typer.Option(
        True,
        "--strict-five-actions/--no-strict-five-actions",
        help="Require exactly five candidate actions per merged row",
    ),
    show_progress: bool = typer.Option(
        True,
        "--show-progress/--no-show-progress",
        help="Display a progress bar while evaluating rows",
    ),
    max_workers: int = typer.Option(8, "--max-workers", min=1, help="Max concurrent verifier-evaluation workers"),
    limit_rows: int | None = typer.Option(None, "--limit-rows", min=1, help="Optional cap on parsed rows to evaluate"),
    enable_langfuse: bool = typer.Option(
        False,
        "--enable-langfuse",
        help='Enable LiteLLM Langfuse tracing by adding "langfuse_otel" to litellm.callbacks',
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output files if they already exist"),
) -> None:
    try:
        summary = evaluate_verifier_action_selection(
            input_jsonl=Path(input_jsonl),
            output_jsonl=Path(output_jsonl),
            output_summary=Path(output_summary) if output_summary else None,
            config_specs=config_specs,
            verifier_types=verifier_types,
            verifier_variants=verifier_variants,
            strict_five_actions=strict_five_actions,
            limit_rows=limit_rows,
            show_progress=show_progress,
            max_workers=max_workers,
            enable_langfuse=enable_langfuse,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = summary.get("counts", {})
    console.print(f"[green]Wrote per-row output:[/green] {summary.get('output_jsonl')}")
    console.print(f"[green]Wrote summary:[/green] {summary.get('output_summary')}")
    if summary.get("langfuse_session_id"):
        console.print(f"[green]Using Langfuse session_id:[/green] {summary['langfuse_session_id']}")
    console.print(
        "rows_considered={rows_considered} rows_written={rows_written} invalid_rows={invalid_rows}".format(
            rows_considered=counts.get("rows_considered", 0),
            rows_written=counts.get("rows_written", 0),
            invalid_rows=counts.get("invalid_rows", 0),
        )
    )
    for verifier_variant, metrics in (summary.get("per_variant") or {}).items():
        console.print(
            "{name}: evaluated={evaluated} gold_picks={gold_picks} accuracy={accuracy:.4f} "
            "skipped={skipped} failed={failed}".format(
                name=verifier_variant,
                evaluated=metrics.get("rows_evaluated", 0),
                gold_picks=metrics.get("gold_pick_count", 0),
                accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
                skipped=metrics.get("rows_skipped", 0),
                failed=metrics.get("rows_failed", 0),
            )
        )
    for verifier_type, metrics in (summary.get("per_verifier") or {}).items():
        console.print(
            "aggregate[{name}]: evaluated={evaluated} gold_picks={gold_picks} accuracy={accuracy:.4f} "
            "skipped={skipped} failed={failed}".format(
                name=verifier_type,
                evaluated=metrics.get("rows_evaluated", 0),
                gold_picks=metrics.get("gold_pick_count", 0),
                accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
                skipped=metrics.get("rows_skipped", 0),
                failed=metrics.get("rows_failed", 0),
            )
        )
    if output_distribution_csv:
        append_predicted_action_distribution(Path(summary["output_jsonl"]), Path(output_distribution_csv))
        console.print(f"[green]Appended predicted action distribution:[/green] {output_distribution_csv}")


if __name__ == "__main__":
    app()

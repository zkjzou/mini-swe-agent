#!/usr/bin/env python3

"""Evaluate verifier gold-action selection accuracy from merged verifier-action rows."""

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.verifier_action_evaluation import evaluate_verifier_action_selection

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


@app.command(help=__doc__)
def main(
    input_jsonl: str = typer.Option(..., "--input-jsonl", help="Merged verifier-action JSONL file"),
    output_jsonl: str = typer.Option(..., "--output-jsonl", help="Per-row evaluation JSONL output file"),
    output_summary: str | None = typer.Option(None, "--output-summary", help="Optional summary JSON output path"),
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
        help="Verifier type(s) to evaluate. Repeat for multiple: llm, reward_model",
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
    limit_rows: int | None = typer.Option(None, "--limit-rows", min=1, help="Optional cap on parsed rows to evaluate"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output files if they already exist"),
) -> None:
    try:
        summary = evaluate_verifier_action_selection(
            input_jsonl=Path(input_jsonl),
            output_jsonl=Path(output_jsonl),
            output_summary=Path(output_summary) if output_summary else None,
            config_specs=config_specs,
            verifier_types=verifier_types,
            strict_five_actions=strict_five_actions,
            limit_rows=limit_rows,
            show_progress=show_progress,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = summary.get("counts", {})
    console.print(f"[green]Wrote per-row output:[/green] {summary.get('output_jsonl')}")
    console.print(f"[green]Wrote summary:[/green] {summary.get('output_summary')}")
    console.print(
        "rows_considered={rows_considered} rows_written={rows_written} invalid_rows={invalid_rows}".format(
            rows_considered=counts.get("rows_considered", 0),
            rows_written=counts.get("rows_written", 0),
            invalid_rows=counts.get("invalid_rows", 0),
        )
    )
    for verifier_type, metrics in (summary.get("per_verifier") or {}).items():
        console.print(
            "{name}: evaluated={evaluated} gold_picks={gold_picks} accuracy={accuracy:.4f} "
            "skipped={skipped} failed={failed}".format(
                name=verifier_type,
                evaluated=metrics.get("rows_evaluated", 0),
                gold_picks=metrics.get("gold_pick_count", 0),
                accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
                skipped=metrics.get("rows_skipped", 0),
                failed=metrics.get("rows_failed", 0),
            )
        )


if __name__ == "__main__":
    app()

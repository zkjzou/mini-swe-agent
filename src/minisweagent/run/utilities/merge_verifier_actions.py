#!/usr/bin/env python3

"""Merge verifier action-sampling datasets from multiple models/runs into one canonical JSONL."""

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.verifier_dataset_merge import merge_verifier_sampling_datasets

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


@app.command(help=__doc__)
def main(
    inputs: list[str] = typer.Option(
        ...,
        "--inputs",
        help="Input file(s) or directory(ies) containing JSONL datasets. Repeat --inputs for multiple.",
    ),
    output_jsonl: str = typer.Option(..., "--output-jsonl", help="Merged output JSONL file path"),
    output_summary: str | None = typer.Option(None, "--output-summary", help="Optional summary JSON output path"),
    dedupe: str = typer.Option("semantic_key", "--dedupe", help="none|exact|semantic_key"),
    conflict_policy: str = typer.Option("keep_first", "--conflict-policy", help="keep_first|keep_last|error"),
    require_gold: bool = typer.Option(True, "--require-gold/--no-require-gold", help="Require gold row per step"),
    sort_rows: bool = typer.Option(
        True,
        "--sort-rows/--no-sort-rows",
        help="Sort merged rows by instance/step/model/sample for deterministic output",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output files if they already exist"),
) -> None:
    try:
        summary = merge_verifier_sampling_datasets(
            input_paths=[Path(path) for path in inputs],
            output_jsonl=Path(output_jsonl),
            output_summary=Path(output_summary) if output_summary else None,
            dedupe=dedupe,  # type: ignore[arg-type]
            conflict_policy=conflict_policy,  # type: ignore[arg-type]
            require_gold=require_gold,
            sort_by_instance_step_model_sample=sort_rows,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Merge failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = summary.get("counts", {})
    console.print(f"[green]Merged rows:[/green] {counts.get('rows_kept', 0)}")
    console.print(f"[green]Output:[/green] {summary.get('output_jsonl')}")


if __name__ == "__main__":
    app()

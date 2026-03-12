#!/usr/bin/env python3

"""Reanalyze verifier prediction rows under a directory and write a combined distribution CSV."""

from pathlib import Path

import csv
import typer
from rich.console import Console

from minisweagent.run.utilities.evaluate_verifier_actions import _collect_predicted_action_distribution_rows

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _find_rows_jsonl(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*_rows.jsonl") if path.is_file())


@app.command(help=__doc__)
def main(
    input_dir: str = typer.Option(..., "--input-dir", help="Directory to scan for *_rows.jsonl files"),
    output_csv: str = typer.Option(..., "--output-csv", help="Combined distribution CSV output path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the output CSV if it already exists"),
) -> None:
    input_dir_path = Path(input_dir)
    if not input_dir_path.is_dir():
        console.print(f"[red]Input directory not found:[/red] {input_dir_path}")
        raise typer.Exit(code=1)

    output_csv_path = Path(output_csv)
    if output_csv_path.exists() and not overwrite:
        console.print(f"[red]Output file already exists:[/red] {output_csv_path}")
        raise typer.Exit(code=1)

    input_jsonls = _find_rows_jsonl(input_dir_path)
    if not input_jsonls:
        console.print(f"[red]No *_rows.jsonl files found under:[/red] {input_dir_path}")
        raise typer.Exit(code=1)

    rows = _collect_predicted_action_distribution_rows(input_jsonls)
    fieldnames = list(rows[0].keys()) if rows else []
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    console.print(f"[green]Scanned rows files:[/green] {len(input_jsonls)}")
    console.print(f"[green]Wrote combined distribution CSV:[/green] {output_csv_path}")


if __name__ == "__main__":
    app()

#!/usr/bin/env python3

"""Sample additional action candidates from verifier models by replaying successful trajectories."""

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.verifier_action_sampling import generate_verifier_sampling_dataset

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


@app.command(help=__doc__)
def main(
    output_json: str = typer.Option(..., "--output-json", help="Path to Docent output.json file"),
    transcripts_dir: str = typer.Option(..., "--transcripts-dir", help="Directory containing transcript JSON files"),
    sampler_config: str = typer.Option(..., "--sampler-config", help="YAML file listing sampler models"),
    output_dir: str = typer.Option(..., "--output-dir", help="Directory to write candidates.jsonl + summary.json"),
    num_samples: int = typer.Option(2, "--num-samples", min=1, help="Samples per model per step"),
    max_workers: int = typer.Option(8, "--max-workers", min=1, help="Max concurrent sampling requests"),
    limit_runs: int | None = typer.Option(None, "--limit-runs", min=1, help="Optional cap on resolved runs"),
    limit_steps_per_run: int | None = typer.Option(
        None, "--limit-steps-per-run", min=1, help="Optional cap on replayed action steps per run"
    ),
    exclude_parallel_tool_calls: bool = typer.Option(
        True,
        "--exclude-parallel-tool-calls/--allow-parallel-tool-calls",
        help="Skip trajectories containing assistant messages with multiple tool calls",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output files if they already exist"),
) -> None:
    try:
        summary = generate_verifier_sampling_dataset(
            output_json_path=Path(output_json),
            transcripts_dir=Path(transcripts_dir),
            sampler_config_path=Path(sampler_config),
            output_dir=Path(output_dir),
            num_samples=num_samples,
            max_workers=max_workers,
            limit_runs=limit_runs,
            limit_steps_per_run=limit_steps_per_run,
            exclude_parallel_tool_call_trajectories=exclude_parallel_tool_calls,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Sampling failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    counts = summary.get("counts", {})
    console.print(f"[green]Wrote:[/green] {summary.get('output_jsonl')}")
    console.print(
        "runs={runs_processed} steps={steps_processed} gold={gold_candidates} sampled={sample_candidates} "
        "failures={sample_failures}".format(
            runs_processed=counts.get("runs_processed", 0),
            steps_processed=counts.get("steps_processed", 0),
            gold_candidates=counts.get("gold_candidates", 0),
            sample_candidates=counts.get("sample_candidates", 0),
            sample_failures=counts.get("sample_failures", 0),
        )
    )


if __name__ == "__main__":
    app()

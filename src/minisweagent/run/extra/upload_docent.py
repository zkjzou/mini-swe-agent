#!/usr/bin/env python3

"""Upload mini-swe-agent trajectories (.traj.json) to Docent."""

from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.docent_upload import upload_docent

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


@app.command(help=__doc__)
def main(
    path: str = typer.Argument(".", help="Trajectory file or directory to search for .traj.json files"),
    collection_name: str | None = typer.Option(
        None,
        "--collection-name",
        "-c",
        help="Collection name to create in Docent (omit if using --collection-id)",
    ),
    collection_id: str | None = typer.Option(
        None,
        "--collection-id",
        help="Existing Docent collection id to upload into",
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        help="Description for a newly created collection",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DOCENT_API_KEY",
        help="Docent API key (defaults to DOCENT_API_KEY)",
    ),
    server_url: str | None = typer.Option(
        None,
        "--server-url",
        help="Self-hosted Docent server URL (optional)",
    ),
    web_url: str | None = typer.Option(
        None,
        "--web-url",
        help="Self-hosted Docent web URL (optional)",
    ),
    evaluation_result: str | None = typer.Option(
        None,
        "--evaluation-result",
        help="Path to evaluation result JSON for resolved scores",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and validate without uploading"),
) -> None:
    try:
        result = upload_docent(
            Path(path),
            collection_name=collection_name,
            collection_id=collection_id,
            collection_description=description,
            api_key=api_key,
            server_url=server_url,
            web_url=web_url,
            evaluation_result_path=Path(evaluation_result) if evaluation_result else None,
            dry_run=dry_run,
            print_fct=console.print,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Upload failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if result.created_collection:
        console.print(f"[green]Created collection:[/green] {result.collection_id}")


if __name__ == "__main__":
    app()

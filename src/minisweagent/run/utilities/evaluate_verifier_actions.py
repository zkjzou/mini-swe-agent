#!/usr/bin/env python3

"""Evaluate verifier gold-action selection accuracy from merged verifier-action rows."""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.verifier_action_evaluation import evaluate_verifier_action_selection

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

_PREFERRED_LABEL_ORDER = (
    "gold",
    "qwen3_coder_next",
    "qwen3_5_instruct",
    "gpt5mini",
    "qwen3_coder",
)
_DISPLAY_LABEL_NAMES = {
    "gold": "gold",
    "qwen3_coder_next": "qwen3-coder-next",
    "qwen3_5_instruct": "qwen3.5-27b",
    "gpt5mini": "gpt5-mini",
    "qwen3_coder": "qwen3-coder-instruct",
}


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _display_label_name(label: str) -> str:
    return _DISPLAY_LABEL_NAMES.get(label, label)


def _ordered_labels(labels: set[str]) -> list[str]:
    preferred_rank = {label: index for index, label in enumerate(_PREFERRED_LABEL_ORDER)}
    return sorted(labels, key=lambda label: (preferred_rank.get(label, len(preferred_rank)), _display_label_name(label)))


def _make_distribution_column_names(labels: list[str]) -> dict[str, tuple[str, str]]:
    columns: dict[str, tuple[str, str]] = {}
    for label in labels:
        key = _display_label_name(label)
        columns[label] = (f"count__{key}", key)
    return columns


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


def _collect_predicted_action_distribution_rows(output_jsonls: list[Path]) -> list[dict[str, str]]:
    parsed_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    parser_failure_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str, str]] = Counter()
    parsed_totals: Counter[tuple[str, str, str]] = Counter()
    parser_failures: Counter[tuple[str, str, str]] = Counter()
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
                selected_label = _selected_label(row)
                key = (verifier_type, verifier_variant, str(output_jsonl))
                totals[key] += 1
                if _has_parser_failure(row):
                    parser_failures[key] += 1
                    parser_failure_counts[key][selected_label] += 1
                else:
                    parsed_counts[key][selected_label] += 1
                    parsed_totals[key] += 1

    labels = _ordered_labels(
        {
            *_PREFERRED_LABEL_ORDER,
            *(
                label
                for label_counts in [*parsed_counts.values(), *parser_failure_counts.values()]
                for label in label_counts
            ),
        }
    )
    label_columns = _make_distribution_column_names(labels)
    front_fieldnames = ["model", "verifier_variant"]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fraction_fieldnames = [label_columns[label][1] for label in labels]
    metadata_fieldnames = [
        "rows_evaluated",
        "rows_non_parser_failed",
        "rows_parser_failed",
        "fraction_parser_failed",
        "timestamp_utc",
        "output_jsonl",
    ]
    count_fieldnames = [label_columns[label][0] for label in labels]
    parser_failure_fraction_fieldnames = [
        column_name
        for label in labels
        for column_name in (f"parser_failure_fraction__{_display_label_name(label)}",)
    ]
    parser_failure_count_fieldnames = [
        column_name
        for label in labels
        for column_name in (f"parser_failure_count__{_display_label_name(label)}",)
    ]
    fieldnames = (
        front_fieldnames
        + fraction_fieldnames
        + metadata_fieldnames
        + count_fieldnames
        + parser_failure_fraction_fieldnames
        + parser_failure_count_fieldnames
    )
    rows: list[dict[str, str]] = []
    all_keys = sorted(set(totals))
    for verifier_type, verifier_variant, output_jsonl in all_keys:
        key = (verifier_type, verifier_variant, output_jsonl)
        total = totals[key]
        parsed_total = parsed_totals[key]
        parser_failed = parser_failures[key]
        row = {
            "model": _display_model_name(Path(output_jsonl), verifier_type),
            "verifier_variant": verifier_variant,
            "rows_evaluated": str(total),
            "rows_non_parser_failed": str(parsed_total),
            "rows_parser_failed": str(parser_failed),
            "fraction_parser_failed": f"{parser_failed / total:.6f}" if total else "0.000000",
            "timestamp_utc": timestamp,
            "output_jsonl": output_jsonl,
        }
        for selected_label, count in sorted(parsed_counts[key].items()):
            count_column, fraction_column = label_columns[selected_label]
            row[count_column] = str(count)
            row[fraction_column] = f"{count / parsed_total:.6f}" if parsed_total else "0.000000"
        for selected_label, count in sorted(parser_failure_counts[key].items()):
            label_key = _display_label_name(selected_label)
            row[f"parser_failure_count__{label_key}"] = str(count)
            row[f"parser_failure_fraction__{label_key}"] = (
                f"{count / parser_failed:.6f}" if parser_failed else "0.000000"
            )
        rows.append({field: row.get(field, "") for field in fieldnames})
    return rows


def append_predicted_action_distribution(output_jsonl: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = [dict(row) for row in reader]
    new_rows = _collect_predicted_action_distribution_rows([output_jsonl])
    new_fieldnames = list(new_rows[0].keys()) if new_rows else []
    fieldnames = list(dict.fromkeys(existing_fieldnames + new_fieldnames))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({field: existing_row.get(field, "") for field in fieldnames})
        for row in new_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _print_metric_summary(prefix: str, metrics: dict[str, object]) -> None:
    evaluated = _safe_int(metrics.get("rows_evaluated"))
    total_cost = _safe_float(metrics.get("total_cost"))
    average_cost = _safe_float(metrics.get("average_cost"))
    if average_cost <= 0.0 and evaluated > 0:
        average_cost = total_cost / evaluated
    console.print(
        "{prefix}: evaluated={evaluated} gold_picks={gold_picks} accuracy={accuracy:.4f} "
        "skipped={skipped} failed={failed} cost=${cost:.4f} avg_cost=${avg_cost:.4f} api_calls={api_calls}".format(
            prefix=prefix,
            evaluated=evaluated,
            gold_picks=_safe_int(metrics.get("gold_pick_count")),
            accuracy=_safe_float(metrics.get("accuracy")),
            skipped=_safe_int(metrics.get("rows_skipped")),
            failed=_safe_int(metrics.get("rows_failed")),
            cost=total_cost,
            avg_cost=average_cost,
            api_calls=_safe_int(metrics.get("total_api_calls")),
        ),
        markup=False,
    )


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
    overall_metrics = summary.get("overall")
    if isinstance(overall_metrics, dict):
        _print_metric_summary("overall", overall_metrics)
    for verifier_variant, metrics in (summary.get("per_variant") or {}).items():
        if isinstance(metrics, dict):
            _print_metric_summary(verifier_variant, metrics)
    for verifier_type, metrics in (summary.get("per_verifier") or {}).items():
        if isinstance(metrics, dict):
            _print_metric_summary(f"aggregate[{verifier_type}]", metrics)
    if output_distribution_csv:
        append_predicted_action_distribution(Path(summary["output_jsonl"]), Path(output_distribution_csv))
        console.print(f"[green]Appended predicted action distribution:[/green] {output_distribution_csv}")


if __name__ == "__main__":
    app()

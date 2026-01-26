#!/usr/bin/env python3
"""Offline rejected action sampling for existing trajectories."""

import json
from pathlib import Path
from typing import Iterable

import typer
import yaml

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.utils.rejected_actions import (
    DEFAULT_ACTION_REGEX,
    RejectedActionSampler,
    extract_action_from_response,
    resolve_output_path,
    write_rejected_actions,
)
from minisweagent.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
DEFAULT_CONFIG = builtin_config_dir / "extra" / "swebench.yaml"


def _trajectory_base_name(path: Path) -> str:
    name = path.name
    if name.endswith(".traj.json"):
        return name[: -len(".traj.json")]
    return path.stem


def _iter_trajectories(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.traj.json"))


# fmt: off
@app.command()
def main(
    path: Path = typer.Argument(..., help="Trajectory file or directory"),
    config: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing rejected action files"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Directory for rejected action files"),
    run_id: str = typer.Option("", "--run-id", help="Optional run identifier appended to output filename"),
) -> None:
    # fmt: on
    """Generate rejected actions for existing trajectories."""
    config_path = get_config_path(config)
    logger.info("Loading config from '%s'", config_path)
    cfg = yaml.safe_load(config_path.read_text())
    sampling_cfg = cfg.get("rejected_action_sampling", {})
    if not sampling_cfg.get("enabled"):
        raise typer.BadParameter("rejected_action_sampling.enabled must be true")
    mode = sampling_cfg.get("mode", "offline")
    if mode not in {"offline", "both"}:
        logger.warning("Rejected action sampling mode '%s' does not include offline", mode)
    model_pool = sampling_cfg.get("model_pool", [])
    if not model_pool:
        raise typer.BadParameter("rejected_action_sampling.model_pool must be set")

    trajectories = list(_iter_trajectories(path))
    if not trajectories:
        raise typer.BadParameter(f"No trajectory files found in '{path}'")

    for traj_path in trajectories:
        traj_data = json.loads(traj_path.read_text())
        messages = traj_data.get("messages", [])
        if not messages:
            logger.warning("Skipping %s (no messages)", traj_path)
            continue

        action_regex = (
            traj_data.get("info", {})
            .get("config", {})
            .get("agent", {})
            .get("action_regex", DEFAULT_ACTION_REGEX)
        )
        expert_model_name = (
            traj_data.get("info", {})
            .get("config", {})
            .get("model", {})
            .get("model_name")
        )
        expert_model_class = traj_data.get("info", {}).get("config", {}).get("model_type")

        base_name = _trajectory_base_name(traj_path)
        output_base = (output_dir or traj_path.parent) / f"{base_name}.rejected.jsonl"
        resolved_output = resolve_output_path(output_base, run_id or sampling_cfg.get("run_id"))
        if resolved_output.exists() and not overwrite:
            raise typer.BadParameter(f"Rejected action file already exists: {resolved_output}")

        sampler = RejectedActionSampler(
            model_pool=model_pool,
            action_regex=action_regex,
            selection_policy=sampling_cfg.get("selection_policy", "round_robin"),
            k_per_step=sampling_cfg.get("k_per_step", 1),
            max_attempts_per_step=sampling_cfg.get("max_attempts_per_step"),
            timeout_s=sampling_cfg.get("timeout_s"),
            expert_model_name=expert_model_name,
            expert_model_class=expert_model_class,
            output_path=output_base,
            overwrite=overwrite,
            run_id=run_id or sampling_cfg.get("run_id"),
            mode="offline",
        )

        records: list[dict] = []
        step_index = 0
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            try:
                expert_action = extract_action_from_response(message.get("content", ""), action_regex)
            except ValueError:
                continue
            prompt_messages = messages[:idx]
            records.extend(
                sampler.sample_records(
                    prompt_messages=prompt_messages,
                    step_index=step_index,
                    expert_action=expert_action,
                )
            )
            step_index += 1

        if not records:
            logger.warning("No rejected actions sampled for %s", traj_path)
            continue

        write_rejected_actions(resolved_output, records, overwrite=overwrite)
        logger.info("Saved %s rejected actions to %s", len(records), resolved_output)


if __name__ == "__main__":
    app()

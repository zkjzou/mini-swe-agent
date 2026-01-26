#!/usr/bin/env python3
"""Replay a trajectory to a target step and run Monte Carlo rollouts from that state."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import typer
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.rollout_actions import (
    FixedActionProvider,
    ModelActionProvider,
    RolloutActionProvider,
    RolloutStepError,
)
from minisweagent.run.extra.utils.trajectory_replay import RegexActionSelector, TrajectoryReplayer
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.trajectory import load_trajectory
from minisweagent.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_SUMMARY_LOCK = threading.Lock()
DEFAULT_ACTION_REGEX = r"```bash\s*\n(.*?)\n```"


class RolloutFailure(Exception):
    pass


def _trajectory_base_name(path: Path) -> str:
    name = path.name
    if name.endswith(".traj.json"):
        return name[: -len(".traj.json")]
    return path.stem


def _iter_trajectories(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.traj.json"))


def _parse_json_arg(raw: str, *, name: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for {name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{name} must be a JSON object")
    return parsed


def _parse_actions(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for --rollout-actions-json: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise typer.BadParameter("--rollout-actions-json must be a JSON array of strings")
    return parsed


def _resolve_agent_config(traj_info: dict[str, Any], fallback_config: dict[str, Any]) -> dict[str, Any]:
    agent_config = traj_info.get("config", {}).get("agent")
    if agent_config:
        return agent_config
    if "agent" in fallback_config:
        return fallback_config["agent"]
    raise RolloutFailure("Agent config missing from trajectory and fallback config")


def _resolve_model_config(
    traj_info: dict[str, Any],
    fallback_config: dict[str, Any],
    *,
    model_name: str | None,
    model_class: str | None,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    model_config = dict(fallback_config.get("model", {}))
    traj_model = traj_info.get("config", {}).get("model")
    if traj_model:
        model_config.update(traj_model)
    if model_name:
        model_config["model_name"] = model_name
    if model_class:
        model_config["model_class"] = model_class
    else:
        traj_model_type = traj_info.get("config", {}).get("model_type")
        if traj_model_type:
            model_config.setdefault("model_class", traj_model_type)
    if "model_name" not in model_config and model_class:
        model_config["model_name"] = model_class
    if model_kwargs:
        model_config.setdefault("model_kwargs", {}).update(model_kwargs)
    return model_config


def _resolve_env_config(
    traj_info: dict[str, Any],
    fallback_config: dict[str, Any],
    *,
    environment_class: str | None,
    environment_overrides: dict[str, Any],
) -> dict[str, Any]:
    env_config = dict(fallback_config.get("environment", {}))
    traj_env = traj_info.get("config", {}).get("environment")
    if traj_env:
        env_config.update(traj_env)
    if environment_class:
        env_config["environment_class"] = environment_class
    else:
        traj_env_type = traj_info.get("config", {}).get("environment_type")
        if traj_env_type:
            env_config.setdefault("environment_class", traj_env_type)
    if environment_overrides:
        env_config.update(environment_overrides)
    if "environment_class" not in env_config:
        env_config["environment_class"] = "local"
    return env_config


def _write_summary(path: Path, record: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SUMMARY_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


# fmt: off
@app.command()
def main(
    trajectory: Path = typer.Argument(..., help="Trajectory file or directory"),
    step: int = typer.Option(1, "--step", help="Target action step index (1-based). Use 0 to skip replay."),
    rollouts: int = typer.Option(1, "--rollouts", help="Number of rollouts per trajectory"),
    rollout_steps: int = typer.Option(1, "--rollout-steps", help="Maximum rollout steps to execute"),
    output_dir: Path = typer.Option(Path("monte_carlo_rollouts"), "--output", help="Output directory"),
    workers: int = typer.Option(1, "--workers", help="Number of worker threads"),
    include_thoughts: bool = typer.Option(
        True,
        "--include-thoughts/--exclude-thoughts",
        help="Include assistant messages from the trajectory in rollout prompts",
    ),
    verify_observations: bool = typer.Option(False, "--verify-observations", help="Check replayed observations match"),
    config: Path = typer.Option(builtin_config_dir / "default.yaml", "-c", "--config", help="Fallback config file"),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Override model class"),
    model_kwargs_json: str = typer.Option("", "--model-kwargs-json", help="JSON object for model kwargs"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Override environment class"),
    environment_config_json: str = typer.Option("", "--environment-config-json", help="JSON object for environment overrides"),
    env_startup_command: str = typer.Option("", "--env-startup-command", help="Command to run before replay"),
    rollout_action: str = typer.Option("", "--rollout-action", help="Fixed action to execute instead of model sampling"),
    rollout_actions_json: str = typer.Option("", "--rollout-actions-json", help="JSON array of fixed actions"),
) -> None:
    # fmt: on
    if rollouts < 1:
        raise typer.BadParameter("--rollouts must be >= 1")
    if rollout_steps < 1:
        raise typer.BadParameter("--rollout-steps must be >= 1")
    if step < 0:
        raise typer.BadParameter("--step must be >= 0")
    if rollout_action and rollout_actions_json:
        raise typer.BadParameter("Use only one of --rollout-action or --rollout-actions-json")

    config_path = get_config_path(config)
    fallback_config = yaml.safe_load(config_path.read_text()) if config_path else {}
    model_kwargs = _parse_json_arg(model_kwargs_json, name="--model-kwargs-json")
    env_overrides = _parse_json_arg(environment_config_json, name="--environment-config-json")

    fixed_actions = _parse_actions(rollout_actions_json)
    if rollout_action:
        fixed_actions = [rollout_action]

    trajectories = list(_iter_trajectories(trajectory))
    if not trajectories:
        raise typer.BadParameter(f"No trajectory files found in '{trajectory}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rollouts.jsonl"

    tasks: list[tuple[Path, int]] = []
    for traj_path in trajectories:
        for rollout_index in range(rollouts):
            tasks.append((traj_path, rollout_index))

    def run_rollout(task: tuple[Path, int]) -> None:
        traj_path, rollout_index = task
        start_time = time.time()
        traj_data = load_trajectory(traj_path)
        messages = traj_data["messages"]
        info = traj_data.get("info", {})

        agent_config = _resolve_agent_config(info, fallback_config)
        model_config = _resolve_model_config(
            info,
            fallback_config,
            model_name=model,
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
        env_config = _resolve_env_config(
            info,
            fallback_config,
            environment_class=environment_class,
            environment_overrides=env_overrides,
        )
        action_regex = agent_config.get("action_regex", DEFAULT_ACTION_REGEX)

        env = get_environment(env_config)
        try:
            if env_startup_command:
                startup_out = env.execute(env_startup_command)
                if startup_out.get("returncode", 0) != 0:
                    raise RolloutFailure(f"Startup command failed: {startup_out}")

            replayer = TrajectoryReplayer(
                messages,
                agent_config,
                env,
                include_thoughts=include_thoughts,
                action_selector=RegexActionSelector(action_regex),
                verify_observations=verify_observations,
            )
            replay_result = replayer.replay_to_step(step)

            model_obj = get_model(config=model_config)
            agent = DefaultAgent(model_obj, env, **agent_config)
            agent.messages = [msg.copy() for msg in replay_result.history_messages]

            action_provider: RolloutActionProvider
            if fixed_actions:
                action_provider = FixedActionProvider(fixed_actions)
            else:
                action_provider = ModelActionProvider()

            action_records: list[dict[str, Any]] = []
            for selected in replay_result.selected_actions:
                action_records.append(
                    {
                        "phase": "replay",
                        "action": selected.action,
                        "source": selected.source,
                        **selected.metadata,
                    }
                )

            exit_status = "RolloutComplete"
            result = ""
            rollout_steps_executed = 0
            try:
                for rollout_step in range(rollout_steps):
                    res = action_provider.execute(agent, step_index=rollout_step)
                    rollout_steps_executed += 1
                    action_records.append(
                        {
                            "phase": "rollout",
                            "action": res.action,
                            "source": res.source,
                            "returncode": res.output.get("returncode") if res.output else None,
                            "output_len": len(res.output.get("output", "")) if res.output else None,
                        }
                    )
            except RolloutStepError as exc:
                exit_status = type(exc.cause).__name__
                result = str(exc.cause)
                if exc.action:
                    action_records.append(
                        {
                            "phase": "rollout",
                            "action": exc.action,
                            "source": "model" if not fixed_actions else "fixed",
                            "returncode": None,
                            "output_len": None,
                        }
                    )
            except Exception as exc:
                exit_status = type(exc).__name__
                result = str(exc)

            rollout_info = {
                "source_trajectory": str(traj_path),
                "source_step": step,
                "include_thoughts": include_thoughts,
                "rollout_index": rollout_index,
                "replayed_steps": replay_result.replayed_steps,
                "total_action_steps": replay_result.total_action_steps,
                "rollout_steps_executed": rollout_steps_executed,
                "rollout_steps_requested": rollout_steps,
                "action_source": "fixed" if fixed_actions else "model",
                "model_name": getattr(model_obj.config, "model_name", None),
                "model_class": f"{model_obj.__class__.__module__}.{model_obj.__class__.__name__}",
                "model_kwargs": model_config.get("model_kwargs", {}),
                "model_cost": getattr(model_obj, "cost", None),
                "model_calls": getattr(model_obj, "n_calls", None),
                "outcome": exit_status,
                "mismatches": replay_result.mismatches,
                "actions": action_records,
                "duration_s": round(time.time() - start_time, 3),
            }

            base_dir = output_dir / _trajectory_base_name(traj_path)
            base_dir.mkdir(parents=True, exist_ok=True)
            rollout_path = base_dir / f"rollout_{rollout_index:04d}.traj.json"
            save_traj(
                agent,
                rollout_path,
                exit_status=exit_status,
                result=result,
                extra_info={"rollout": rollout_info},
            )

            _write_summary(
                summary_path,
                {
                    "trajectory": str(traj_path),
                    "rollout_index": rollout_index,
                    "step": step,
                    "include_thoughts": include_thoughts,
                    "action_source": "fixed" if fixed_actions else "model",
                    "model_name": rollout_info["model_name"],
                    "model_class": rollout_info["model_class"],
                    "replayed_steps": replay_result.replayed_steps,
                    "rollout_steps_executed": rollout_steps_executed,
                    "outcome": exit_status,
                    "output_path": str(rollout_path),
                },
            )
        finally:
            if hasattr(env, "stop"):
                env.stop()
            elif hasattr(env, "cleanup"):
                env.cleanup()

    def handle_futures(futures: dict[concurrent.futures.Future, tuple[Path, int]]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                traj_path, rollout_index = futures[future]
                logger.error("Rollout failed for %s (%s): %s", traj_path, rollout_index, exc, exc_info=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_rollout, task): task for task in tasks}
        handle_futures(futures)


if __name__ == "__main__":
    app()

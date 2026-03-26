#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import logging
import random
import re
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path

import litellm
import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utilities.evaluation_client import auto_submit_swebench_predictions
from minisweagent.utils.langfuse import (
    attach_langfuse_session_metadata as _attach_shared_langfuse_session_metadata,
    enable_langfuse_tracing as _enable_shared_langfuse_tracing,
    make_langfuse_session_id as _make_shared_langfuse_session_id,
)
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
_EVAL_SERVER_SUBSET_MAPPING = {
    "verified": "swe-bench_verified",
    "smith": "swe-smith",
    "swe-bench_verified": "swe-bench_verified",
    "swe-smith": "swe-smith",
}


def _iter_seeded_output_paths(output: str, num_seeds: int) -> list[tuple[int | None, Path]]:
    if num_seeds == 1:
        return [(None, Path(output))]
    return [(seed, Path(f"{output}_{seed}")) for seed in range(1, num_seeds + 1)]


def _apply_run_seed(config: dict, *, seed: int | None) -> dict:
    seeded_config = deepcopy(config)
    if seed is None:
        return seeded_config

    model_config = seeded_config.setdefault("model", {})
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["seed"] = seed

    verifier_config = seeded_config.get("agent", {}).get("verifier")
    if isinstance(verifier_config, dict):
        verifier_model = verifier_config.get("model")
        if isinstance(verifier_model, dict):
            verifier_model_kwargs = verifier_model.setdefault("model_kwargs", {})
            verifier_model_kwargs["seed"] = seed

    return seeded_config


def _build_run_config(
    *,
    config_spec: list[str],
    environment_class: str | None,
    model: str | None,
    model_class: str | None,
    seed: int | None,
) -> dict:
    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)
    config = _resolve_profiled_model_config(config)
    return _apply_run_seed(config, seed=seed)


def _with_run_log_handler(output_path: Path) -> logging.FileHandler:
    logger_instance = logging.getLogger("minisweagent")
    previous_handler_count = len(logger_instance.handlers)
    add_file_handler(output_path / "minisweagent.log")
    return logger_instance.handlers[previous_handler_count]


def _remove_log_handler(handler: logging.Handler) -> None:
    logger.removeHandler(handler)
    handler.close()


def _enable_langfuse_tracing() -> None:
    _enable_shared_langfuse_tracing()


def _make_langfuse_session_id(*, subset: str, split: str, output_path: Path) -> str:
    return _make_shared_langfuse_session_id(prefix="swebench", output_path=output_path, parts=[subset, split])


def _attach_langfuse_session_metadata(config: dict, *, session_id: str) -> None:
    _attach_shared_langfuse_session_metadata(config, session_id=session_id)


def _resolve_eval_server_subset(subset: str) -> str:
    return _EVAL_SERVER_SUBSET_MAPPING.get(subset, subset)


def _resolve_profiled_model_config(config: dict) -> dict:
    """Resolve optional actor/verifier profile selectors in a SWE-bench config."""
    resolved = recursive_merge(config)
    profiles = resolved.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise ValueError("Invalid config: 'profiles' must be a mapping.")

    model_profiles = profiles.get("model_profiles", {}) or {}
    verifier_prompt_profiles = profiles.get("verifier_prompts", {}) or {}

    if not isinstance(model_profiles, dict):
        raise ValueError("Invalid config: 'profiles.model_profiles' must be a mapping.")
    if not isinstance(verifier_prompt_profiles, dict):
        raise ValueError("Invalid config: 'profiles.verifier_prompts' must be a mapping.")

    agent_model_profile = resolved.get("agent_model_profile")
    if not agent_model_profile:
        # Backward compatibility alias for older configs/scripts.
        agent_model_profile = resolved.get("model_profile")
    if agent_model_profile:
        if agent_model_profile not in model_profiles:
            available = ", ".join(sorted(model_profiles)) or "<none>"
            raise ValueError(
                f"Unknown agent_model_profile '{agent_model_profile}'. Available profiles: {available}"
            )
        actor_profile = model_profiles[agent_model_profile]
        if not isinstance(actor_profile, dict):
            raise ValueError(f"Invalid model profile '{agent_model_profile}': expected a mapping.")
        existing_model_config = resolved.get("model", {}) or {}
        if not isinstance(existing_model_config, dict):
            raise ValueError("Invalid config: 'model' must be a mapping.")
        # Keep explicit model overrides from config/CLI on top of profile defaults.
        resolved["model"] = recursive_merge(actor_profile, existing_model_config)

    verifier_model_profile = resolved.get("verifier_model_profile")
    verifier_prompt_profile = resolved.get("verifier_prompt_profile")
    if verifier_model_profile or verifier_prompt_profile:
        agent_config = resolved.get("agent", {}) or {}
        if not isinstance(agent_config, dict):
            raise ValueError("Invalid config: 'agent' must be a mapping.")
        verifier_config = agent_config.get("verifier", {}) or {}
        if not isinstance(verifier_config, dict):
            raise ValueError("Invalid config: 'agent.verifier' must be a mapping.")

        if verifier_model_profile:
            if verifier_model_profile not in model_profiles:
                available = ", ".join(sorted(model_profiles)) or "<none>"
                raise ValueError(
                    f"Unknown verifier_model_profile '{verifier_model_profile}'. Available profiles: {available}"
                )
            verifier_model = model_profiles[verifier_model_profile]
            if not isinstance(verifier_model, dict):
                raise ValueError(f"Invalid model profile '{verifier_model_profile}': expected a mapping.")
            existing_verifier_model = verifier_config.get("model", {}) or {}
            if not isinstance(existing_verifier_model, dict):
                raise ValueError("Invalid config: 'agent.verifier.model' must be a mapping.")
            # Keep explicit verifier model overrides from config/CLI on top of profile defaults.
            verifier_config["model"] = recursive_merge(verifier_model, existing_verifier_model)

        if verifier_prompt_profile:
            if verifier_prompt_profile not in verifier_prompt_profiles:
                available = ", ".join(sorted(verifier_prompt_profiles)) or "<none>"
                raise ValueError(
                    f"Unknown verifier_prompt_profile '{verifier_prompt_profile}'. Available profiles: {available}"
                )
            prompt_profile = verifier_prompt_profiles[verifier_prompt_profile]
            prompt_name = None
            prompt_dir = None
            if isinstance(prompt_profile, str):
                prompt_name = prompt_profile
            elif isinstance(prompt_profile, dict):
                prompt_name = prompt_profile.get("prompt_name")
                prompt_dir = prompt_profile.get("prompt_dir")
            else:
                raise ValueError(
                    f"Invalid verifier prompt profile '{verifier_prompt_profile}': expected string or mapping."
                )
            if not isinstance(prompt_name, str) or not prompt_name.strip():
                raise ValueError(
                    f"Invalid verifier prompt profile '{verifier_prompt_profile}': expected a non-empty prompt name."
                )
            if prompt_dir is not None and (not isinstance(prompt_dir, str) or not prompt_dir.strip()):
                raise ValueError(
                    f"Invalid verifier prompt profile '{verifier_prompt_profile}': expected a non-empty prompt_dir."
                )

            existing_prompt_name = verifier_config.get("prompt_name")
            if not isinstance(existing_prompt_name, str) or not existing_prompt_name.strip():
                verifier_config["prompt_name"] = prompt_name

            if isinstance(prompt_dir, str):
                existing_prompt_dir = verifier_config.get("prompt_dir")
                if not isinstance(existing_prompt_dir, str) or not existing_prompt_dir.strip():
                    verifier_config["prompt_dir"] = prompt_dir

        agent_config["verifier"] = verifier_config
        resolved["agent"] = agent_config

    # These are config-construction helpers, not runtime model/agent config keys.
    for helper_key in ("profiles", "model_profile", "verifier_model_profile", "verifier_prompt_profile"):
        resolved.pop(helper_key, None)
    resolved.pop("agent_model_profile", None)
    return resolved


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id
        if instance_id:
            self.extra_template_vars.setdefault("instance_id", instance_id)
        self._display_step = 0

    def step(self) -> dict:
        """Override step to provide progress updates."""
        # Display step attempts (outer agent loop iterations): this avoids inflation from
        # candidate sampling and still advances on retry loops.
        self._display_step += 1
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self._display_step:3d} (${self.cost:.2f})",
        )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def is_error_trajectory(traj_path: Path) -> bool:
    """Return True if trajectory is missing/invalid or indicates an error exit status."""
    if not traj_path.exists():
        return True
    try:
        traj_data = json.loads(traj_path.read_text())
    except json.JSONDecodeError:
        return True
    exit_status = traj_data.get("info", {}).get("exit_status")
    return exit_status not in {"Submitted", "LimitsExceeded"}


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def _run_swebench_batch(
    *,
    subset: str,
    split: str,
    slice_spec: str,
    filter_spec: str,
    shuffle: bool,
    output_path: Path,
    workers: int,
    redo_existing: bool,
    redo_errors: bool,
    config: dict,
    enable_langfuse: bool,
    auto_eval: bool,
    eval_server_url: str,
    eval_run_id: str | None,
    eval_rerun: bool,
    eval_timeout: int | None,
    eval_max_workers: int | None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    log_handler = _with_run_log_handler(output_path)

    try:
        if enable_langfuse is True:
            session_id = _make_langfuse_session_id(subset=subset, split=split, output_path=output_path)
            _attach_langfuse_session_metadata(config, session_id=session_id)
            logger.info("Using Langfuse session_id=%s", session_id)

        from datasets import load_dataset

        dataset_path = DATASET_MAPPING.get(subset, subset)
        logger.info(f"Loading dataset {dataset_path}, split {split}...")
        instances = list(load_dataset(dataset_path, split=split))

        instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
        if redo_existing and redo_errors:
            logger.info("--redo-existing overrides --redo-errors; running all instances.")
        if not redo_existing and (output_path / "preds.json").exists():
            existing_instances = set(json.loads((output_path / "preds.json").read_text()).keys())
            if redo_errors:
                candidate_ids = {instance["instance_id"] for instance in instances}
                existing_instances &= candidate_ids
                error_instances = {
                    instance_id
                    for instance_id in existing_instances
                    if is_error_trajectory(output_path / instance_id / f"{instance_id}.traj.json")
                }
                skip_instances = existing_instances - error_instances
                if error_instances:
                    logger.info(f"Redoing {len(error_instances)} instances with error trajectories")
                if skip_instances:
                    logger.info(f"Skipping {len(skip_instances)} existing instances")
                instances = [instance for instance in instances if instance["instance_id"] not in skip_instances]
            else:
                logger.info(f"Skipping {len(existing_instances)} existing instances")
                instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
        logger.info(f"Running on {len(instances)} instances...")

        progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

        def process_futures(futures: dict[concurrent.futures.Future, str]):
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as e:
                    instance_id = futures[future]
                    logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                    progress_manager.on_uncaught_exception(instance_id, e)

        with Live(progress_manager.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                        "instance_id"
                    ]
                    for instance in instances
                }
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)

        preds_path = output_path / "preds.json"
        if auto_eval and preds_path.exists():
            eval_subset = _resolve_eval_server_subset(subset)
            try:
                response, upload_path, metadata_path = auto_submit_swebench_predictions(
                    preds_path=preds_path,
                    output_dir=output_path,
                    subset=eval_subset,
                    split=split,
                    server_url=eval_server_url,
                    run_id=eval_run_id,
                    rerun=eval_rerun,
                    timeout=eval_timeout,
                    max_workers=eval_max_workers,
                )
                logger.info(
                    "Queued evaluation job_id=%s run_id=%s status=%s position_in_queue=%s upload=%s metadata=%s server=%s",
                    response.get("job_id"),
                    response.get("run_id"),
                    response.get("status"),
                    response.get("position_in_queue"),
                    upload_path,
                    metadata_path,
                    eval_server_url,
                )
            except Exception as exc:
                logger.error("Failed to auto-submit predictions to evaluation server: %s", exc, exc_info=True)
        elif auto_eval:
            logger.info("Skipping evaluation-server submission because no preds.json was produced.")
    finally:
        _remove_log_handler(log_handler)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    num_seeds: int = typer.Option(1, "--num-seeds", min=1, help="Repeat the full batch run across seeds 1..N", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    redo_errors: bool = typer.Option(False, "--redo-errors", help="Redo existing instances with error trajectories", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    enable_langfuse: bool = typer.Option(
        False,
        "--enable-langfuse",
        help='Enable LiteLLM Langfuse tracing by adding "langfuse_otel" to litellm.callbacks',
        rich_help_panel="Advanced",
    ),
    auto_eval: bool = typer.Option(
        True,
        "--auto-eval/--no-auto-eval",
        help="Submit preds.json to the evaluation server after the batch run completes",
        rich_help_panel="Advanced",
    ),
    eval_server_url: str = typer.Option(
        "http://laplace.eecs.umich.edu:8000",
        "--eval-server-url",
        help="Evaluation server URL used for post-run prediction submission",
        rich_help_panel="Advanced",
    ),
    eval_run_id: str | None = typer.Option(
        None,
        "--eval-run-id",
        help="Stable evaluation-server run_id override for cache reuse across reruns",
        rich_help_panel="Advanced",
    ),
    eval_rerun: bool = typer.Option(
        False,
        "--eval-rerun/--no-eval-rerun",
        help="Force a fresh evaluation submission instead of reusing an existing evaluation-server run_id",
        rich_help_panel="Advanced",
    ),
    eval_timeout: int | None = typer.Option(
        None,
        "--eval-timeout",
        help="Optional per-instance timeout sent to the evaluation server",
        rich_help_panel="Advanced",
    ),
    eval_max_workers: int | None = typer.Option(
        None,
        "--eval-max-workers",
        help="Optional worker count sent to the evaluation server",
        rich_help_panel="Advanced",
    ),
) -> None:
    # fmt: on
    num_seeds = int(getattr(num_seeds, "default", num_seeds))
    auto_eval = bool(getattr(auto_eval, "default", auto_eval))
    eval_server_url = str(getattr(eval_server_url, "default", eval_server_url))
    eval_run_id = getattr(eval_run_id, "default", eval_run_id)
    eval_rerun = bool(getattr(eval_rerun, "default", eval_rerun))
    eval_timeout = getattr(eval_timeout, "default", eval_timeout)
    eval_max_workers = getattr(eval_max_workers, "default", eval_max_workers)

    if enable_langfuse is True:
        _enable_langfuse_tracing()
        logger.info('Enabled LiteLLM Langfuse tracing via litellm.callbacks=["langfuse_otel"]')

    for seed, output_path in _iter_seeded_output_paths(output, num_seeds):
        config = _build_run_config(
            config_spec=config_spec,
            environment_class=environment_class,
            model=model,
            model_class=model_class,
            seed=seed,
        )
        _run_swebench_batch(
            subset=subset,
            split=split,
            slice_spec=slice_spec,
            filter_spec=filter_spec,
            shuffle=shuffle,
            output_path=output_path,
            workers=workers,
            redo_existing=redo_existing,
            redo_errors=redo_errors,
            config=config,
            enable_langfuse=enable_langfuse,
            auto_eval=auto_eval,
            eval_server_url=eval_server_url,
            eval_run_id=eval_run_id,
            eval_rerun=eval_rerun,
            eval_timeout=eval_timeout,
            eval_max_workers=eval_max_workers,
        )


if __name__ == "__main__":
    app()

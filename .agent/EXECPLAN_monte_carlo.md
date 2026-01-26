# Monte Carlo Rollouts from Saved Trajectories

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository includes `.agent/PLANS.md`, which defines mandatory ExecPlan requirements. This document must be maintained in accordance with `.agent/PLANS.md`.

## Purpose / Big Picture

The goal is to let a user take an existing saved trajectory, reproduce the execution environment up to a chosen step, and then sample multiple alternative rollouts from that exact state. After this change, a user can point to a `.traj.json` file, select a step index, choose whether to include the assistant’s “thought” text from the trajectory in the rollout prompt, and run Monte Carlo rollouts that each execute a candidate action (or a small sequence of actions) while recording rich metadata such as model identity, sampling parameters, outcome, and steps taken. The user can see it working by running a new `mini-extra` subcommand that produces a directory of rollout trajectory files and a summarized metadata file. The rollout logic is structured around a pluggable action-selection layer so that later, if trajectories include both expert and rejected actions, selecting the rejected action for rollouts is a small, localized change.

## Progress

- [x] (2026-01-24 00:00Z) ExecPlan drafted with repository context, design decisions, and acceptance criteria.
- [x] (2026-01-25 00:10Z) Implement trajectory loading, step segmentation, message-history filtering (include/exclude thoughts), and replay-to-step logic.
- [x] (2026-01-25 00:15Z) Implement Monte Carlo rollout runner and CLI integration.
- [x] (2026-01-25 00:18Z) Add tests and update docs for the new command.
- [ ] (2026-01-25 00:19Z) Validate end-to-end with sample trajectories and parallel rollouts (tests blocked by missing pytest).

## Surprises & Discoveries

- Observation: `pytest` is not available in the environment, so the new tests could not be executed.
  Evidence: `/bin/bash: pytest: command not found`

## Decision Log

- Decision: Define “step” as one assistant action plus its subsequent user observation, using the same step grouping logic as the Textual UI.
  Rationale: This is already the mental model used by the project’s inspector UI, and aligns with how trajectories are displayed to humans.
  Date/Author: 2026-01-24 / Codex

- Decision: Create a new `mini-extra` subcommand rather than modify the default `mini` command.
  Rationale: Monte Carlo rollouts are an advanced workflow and fit the existing “extra” command suite.
  Date/Author: 2026-01-24 / Codex

- Decision: Store rollout metadata inside each saved rollout trajectory under `info.rollout` and also emit a summary JSONL file.
  Rationale: Embedding metadata keeps each trajectory self-describing; a separate summary supports fast analysis across many rollouts.
  Date/Author: 2026-01-24 / Codex

- Decision: Add a rollout option to include or exclude assistant “thought” content from the replayed message history.
  Rationale: Users may want to avoid exposing chain-of-thought text while still reproducing the environment and continuing from the same state.
  Date/Author: 2026-01-24 / Codex

- Decision: Introduce a small action-selection abstraction that can choose which assistant action to replay or roll out.
  Rationale: This isolates future support for trajectories that include both expert and rejected actions without redesigning the runner.
  Date/Author: 2026-01-24 / Codex

- Decision: Add a fixed-action rollout provider to allow explicit actions and ease future rejected-action rollouts.
  Rationale: This keeps the rollout loop modular and makes it straightforward to plug in alternative action sources later.
  Date/Author: 2026-01-25 / Codex

## Outcomes & Retrospective

No outcomes yet; this plan has not been implemented.

## Context and Orientation

Trajectories are saved by `src/minisweagent/run/utils/save.py` using the format `{"info": ..., "messages": ..., "trajectory_format": "mini-swe-agent-1"}`. Some older or test fixtures use a simpler format that is just a list of messages without the `info` wrapper. Messages are dictionaries with at least `role` and `content` fields. “Thoughts” in this plan refer to the assistant message content stored in the trajectory file; depending on the model or prompt, this may include reasoning text alongside the action.

A “step” in mini-swe-agent corresponds to one assistant response and its associated user observation; the Textual UI groups messages into steps using `_messages_to_steps` in `src/minisweagent/agents/interactive_textual.py`. The default agent flow is implemented in `src/minisweagent/agents/default.py`, with action parsing governed by `action_regex` from agent config and action execution via an `Environment` object. Environment implementations live in `src/minisweagent/environments/` and are instantiated through `src/minisweagent/environments/__init__.py`.

The extra CLI entry point is `src/minisweagent/run/mini_extra.py`, which wires subcommands like `inspect`, `swebench`, and `github-issue`. The SWE-bench runner (`src/minisweagent/run/extra/swebench.py`) demonstrates how to run many jobs concurrently using a thread pool.

The new feature will add a Monte Carlo rollout command that reads trajectories (including those outside the repo, such as the example `/scratch/.../astropy__astropy-7166.traj.json`), replays actions to a target step to reproduce environment state, and then runs multiple sampled rollouts, saving each rollout trajectory and metadata. The command will allow filtering the replayed message history to exclude assistant thoughts while keeping user observations, and it will structure action selection so future trajectories with multiple action variants can be handled by swapping a selector.

## Plan of Work

First, add a small utility module to normalize trajectory files and split messages into steps. Place it at `src/minisweagent/run/utils/trajectory.py` so it sits alongside `save.py` and is reusable by other runners. This module should load either a list-format trajectory or a dict-format trajectory with a `messages` field, returning a normalized structure that includes `messages`, `info`, and `trajectory_format` where available. It should also include a `messages_to_steps` function that mirrors the existing step grouping behavior; consider moving or duplicating the logic from `_messages_to_steps` to avoid UI-only dependencies in the new runner. Add a helper that builds the rollout message history with an `include_thoughts` flag; when false, omit assistant messages from the history and keep user/system messages so the model receives only non-thought context.

Next, implement a replay component in a new module such as `src/minisweagent/run/extra/utils/trajectory_replay.py`. This component should accept: the normalized messages, an agent configuration (for `action_regex` and templates), and an environment instance. It should construct a “replay plan” by extracting assistant actions from each step using the same regex as `DefaultAgent.parse_action`, but do so through an action-selection abstraction (for example an `ActionSelector` or `ActionProvider`) that can later choose rejected actions if present. The replay method should iterate through steps until the target step is reached, execute each action in the environment, optionally compare the resulting observation with the saved user message (if present), and track mismatches as metadata. It should also return the filtered message history up to the target step so that rollouts can continue from that exact conversation context with or without assistant thoughts.

Then, implement the Monte Carlo rollout runner in a new CLI module, for example `src/minisweagent/run/extra/monte_carlo.py`. This command should accept a trajectory file or directory, a target step index, a number of rollouts per trajectory, and a maximum number of rollout steps (default one step). It should also accept an explicit `--include-thoughts/--exclude-thoughts` argument that controls whether assistant messages from the trajectory are included in the rollout prompt history. It should accept model selection overrides (`--model`, `--model-class`, `--model-kwargs-json` for passing sampling params like temperature/top_p/seed), and environment overrides (`--environment-class`, `--environment-config-json`, and an optional `--env-startup-command` for SWE-bench-like setups that are not stored in the trajectory). For each rollout, it should instantiate a fresh environment, replay to the target step, and then run the rollout by calling agent `step()` repeatedly, collecting the resulting observations, exit status, cost, and timing. The action-selection abstraction should be threaded through so that later a CLI flag (for example `--action-source rejected`) can be added with minimal changes.

To address efficiency, use a thread pool (patterned after `run/extra/swebench.py`) to run multiple rollouts concurrently. Pre-parse the trajectory into a replay plan once per file, and reuse that plan for each rollout instance to avoid repeated regex parsing. This provides a straightforward speedup without deep environment snapshotting. Keep each rollout isolated by using a new environment instance per rollout; when done, ensure cleanup by calling `stop()` or `cleanup()` if the environment exposes those methods.

Add metadata recording that captures at least: source trajectory path, source step index, rollout index, model name/class and sampling parameters, whether assistant thoughts were included, number of replayed steps, number of rollout steps executed, outcome (Submitted, LimitsExceeded, FormatError, ExecutionTimeoutError, or other exception names), and per-action labels such as return code and output length. Embed this in the saved rollout trajectory under `info.rollout`, and also write a summary JSONL file (one line per rollout) to the output directory for easy analysis.

Wire the new command into `src/minisweagent/run/mini_extra.py` by adding it to the `subcommands` list with an alias such as `monte-carlo` and `mc`. Update or add docs in `docs/` (for example a short page under usage) to explain expected inputs, step indexing, and output structure. Keep documentation concise, and include a minimal example invocation.

Finally, add tests under `tests/run/` to validate: loading list-format and dict-format trajectories; replaying to a target step in a local environment; capturing metadata; include/exclude-thoughts behavior (confirm that assistant messages are present or absent in the rollout prompt history); and deterministic rollouts using `DeterministicModel` with provided outputs. The tests should use temporary directories and local shell commands that are safe and idempotent, such as `echo` or creating files inside a temp directory.

## Concrete Steps

Work from the repository root `/home/zkjzou/SWE-PRM/mini-swe-agent`. Create a new branch for this feature:

    git switch -c feat/monte-carlo-rollouts

Add the new utility and runner modules, then wire the CLI and tests. When adding the CLI command, update `src/minisweagent/run/mini_extra.py` to include the new module and alias. Use the same Typer patterns as other `mini-extra` commands.

When implementing the replay component, ensure it takes an explicit target step index that is clearly defined as 1-based in the CLI, and convert to 0-based internally. Ensure the CLI exposes a boolean `--include-thoughts/--exclude-thoughts` argument that controls whether assistant messages are passed into the rollout prompt. If a trajectory does not include `info.config`, require the user to supply a config path or JSON overrides; make this explicit in error messages.

## Validation and Acceptance

Run unit tests for the new functionality and ensure they fail before the change and pass after:

    pytest -n auto tests/run/test_monte_carlo.py

Run a local smoke test using a small trajectory (for example `tests/test_data/local.traj.json`) and deterministic model outputs:

    mini-extra monte-carlo --trajectory tests/test_data/local.traj.json --step 1 --rollouts 2 --rollout-steps 1 --exclude-thoughts --model-class deterministic --model-kwargs-json '{"outputs":["Step 1\n```bash\necho ok\n```"]}' --output /tmp/mc_out

Acceptance is met when the command produces two rollout trajectories in `/tmp/mc_out`, each with `info.rollout` metadata (including whether thoughts were included), and a summary JSONL file listing the rollouts with correct model and outcome fields. The environment should have executed the replayed commands up to the requested step before the rollout actions run.

## Idempotence and Recovery

All steps are additive and safe to rerun. If output directories already exist, the command should either create a new timestamped subdirectory or cleanly overwrite only per-rollout files for the same rollout index; pick one behavior and document it in the CLI help. If replay fails for a trajectory, the runner should record the failure in the summary file and continue with other rollouts. Environment cleanup should be attempted even on exceptions to avoid leaking containers.

## Artifacts and Notes

Expected rollout summary record format (example fields; actual content defined in implementation):

    {"trajectory_path":".../astropy__astropy-7166.traj.json","step":12,"rollout_index":0,"model_name":"gpt-4.1","model_class":"minisweagent.models.litellm_model.LitellmModel","sampling":{"temperature":0.7,"top_p":0.9,"seed":123},"replayed_steps":12,"rollout_steps":1,"outcome":"Submitted","cost":0.0042,"action_returncode":0,"action_output_len":182}

## Interfaces and Dependencies

Add a small trajectory utility module in `src/minisweagent/run/utils/trajectory.py` with functions like:

    def load_trajectory(path: Path) -> dict[str, Any]:
        """Return a normalized dict containing at least 'messages' and possibly 'info' and 'trajectory_format'."""

    def messages_to_steps(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group messages into steps where each step ends with a user message."""

    def build_message_history(messages: list[dict[str, Any]], *, include_thoughts: bool) -> list[dict[str, Any]]:
        """Return a filtered message history for rollouts; omit assistant messages when include_thoughts is False."""

Add a replay helper in `src/minisweagent/run/extra/utils/trajectory_replay.py` with a minimal interface:

    class ActionSelector(Protocol):
        def select_action(self, step_messages: list[dict[str, Any]]) -> str:
            """Return the action string to replay for this step, e.g., expert or rejected."""

    class TrajectoryReplayer:
        def __init__(
            self,
            messages: list[dict],
            agent_config: dict,
            env: Environment,
            *,
            include_thoughts: bool,
            action_selector: ActionSelector,
            verify_observations: bool = False,
        ): ...
        def replay_to_step(self, target_step: int) -> dict[str, Any]:
            """Execute actions up to target_step, return history and mismatch metadata."""

Implement the CLI in `src/minisweagent/run/extra/monte_carlo.py` using Typer, and update `src/minisweagent/run/mini_extra.py` to expose it. Reuse `get_model` and `get_environment` where possible, but allow full import-path overrides using `model_class` and `environment_class`. Use standard library concurrency (`concurrent.futures.ThreadPoolExecutor`) for parallel rollouts. Avoid new external dependencies. Thread the `include_thoughts` flag and a default `ActionSelector` implementation through the runner so future support for rejected actions is localized.

Note: Initial creation of ExecPlan per user request to save in `.agent/`.
Update: Added include/exclude thoughts CLI argument, message-history filtering, and an action-selection abstraction to future-proof rejected-action rollouts per user request.
Update: Implemented trajectory utilities, replay/rollout modules, CLI wiring, docs, and tests; recorded pytest unavailability during validation.

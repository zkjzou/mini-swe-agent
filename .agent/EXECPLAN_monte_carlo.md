# Monte Carlo Rollouts from Merged Verifier Rows

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository includes `.agent/PLANS.md`, which defines mandatory ExecPlan requirements. This document must be maintained in accordance with `.agent/PLANS.md`.

## Purpose / Big Picture

The goal is to let a user take a merged verifier-action dataset row from `merged_grouped_latest.jsonl`, rebuild the SWE-bench environment state by replaying the exact logged tool-call prefix, and then branch on every candidate action at that step. After this change, a user can run a new `mini-extra monte-carlo-rollout` command that samples `n` full continuations per candidate action, saves a rollout trajectory for each sample, and emits machine-readable summary files for later PRM analysis. The visible proof is a directory of rollout trajectory files plus `results.jsonl` and `summary.json`.

## Progress

- [x] (2026-03-08 17:45Z) Reworked the ExecPlan from saved `.traj.json` replay to merged-row branching based on `history_trajectory` and `actions` from `merged_grouped_latest.jsonl`.
- [x] (2026-03-08 18:05Z) Implemented replay helpers in `src/minisweagent/run/extra/utils/trajectory_replay.py` for task extraction, live prefix replay, and forced candidate assistant message synthesis.
- [x] (2026-03-08 18:15Z) Implemented the `mini-extra monte-carlo-rollout` command in `src/minisweagent/run/extra/monte_carlo.py` and wired it into `src/minisweagent/run/utilities/mini_extra.py`.
- [x] (2026-03-08 18:25Z) Added targeted tests in `tests/run/test_monte_carlo_rollout.py` covering replay, branch message construction, end-to-end rollout generation, CLI invocation, dispatcher wiring, and rollout prediction export.
- [x] (2026-03-08 18:45Z) Added `--redo-existing` and `--redo-errors` semantics plus tests so reruns can skip or selectively recompute existing rollout tasks.
- [x] (2026-03-08 19:05Z) Added SWE-bench-style live progress reporting so rollout tasks show instance/step/action status during replay and continuation.
- [ ] (2026-03-08 18:25Z) Documentation page for the new command remains to be written if user-facing docs are desired.

## Surprises & Discoveries

- Observation: The merged verifier rows already contain the exact replay prefix in `history_trajectory`, so no transcript lookup is needed for v1.
  Evidence: Sample rows show `system`, `user`, `assistant`, and `tool` messages directly in `history_trajectory`.

- Observation: The logged history uses tool-calling message format (`assistant` with `tool_calls`, `tool` for observations), which matches current tool-calling model runtime behavior.
  Evidence: `src/minisweagent/models/utils/actions_toolcall.py` emits `tool` role observations when `tool_call_id` is present.

- Observation: The observed merged dataset shape uses at most one tool call per assistant step, which allows the first implementation to reject parallel replay safely instead of supporting it partially.
  Evidence: A scan over sampled rows found `max_tool_calls == 1` in `history_trajectory` assistant messages.

## Decision Log

- Decision: Use `merged_grouped_latest.jsonl` rows, not saved `.traj.json` files, as the primary Monte Carlo source.
  Rationale: The user explicitly wants branching over verifier candidates, and those candidates already live in merged rows with `actions` and `history_trajectory`.
  Date/Author: 2026-03-08 / Codex

- Decision: Rebuild the live seeded history by replaying recorded commands and appending fresh observation messages, instead of trusting recorded tool outputs.
  Rationale: The user requirement is exact command replay even if the new tool output differs. Using live observations keeps the resumed conversation aligned with the actual environment state.
  Date/Author: 2026-03-08 / Codex

- Decision: Resume continuation with the normal agent loop from the resolved SWE-bench config after forcing the branch action.
  Rationale: This preserves the exact downstream actor/verifier policy under study and isolates the intervention to the chosen branch point.
  Date/Author: 2026-03-08 / Codex

- Decision: Implement the reusable replay logic under `src/minisweagent/run/extra/utils/trajectory_replay.py` and expose the user command from `src/minisweagent/run/extra/monte_carlo.py`.
  Rationale: The user asked for implementation under `run/extra`, but the existing `mini-extra` dispatcher can import subcommands from any module path.
  Date/Author: 2026-03-08 / Codex

- Decision: Save one full rollout trajectory per `(row, action, sample)` and emit a compact `results.jsonl` plus `summary.json`.
  Rationale: The trajectories are needed for qualitative inspection, while the JSONL/summary outputs are needed for aggregate experiment analysis.
  Date/Author: 2026-03-08 / Codex

## Outcomes & Retrospective

The merged-row Monte Carlo runner is implemented and covered by focused tests. A user can now branch over all candidate actions at a chosen verifier step, sample multiple continuations per candidate, and analyze outcomes through saved trajectories and result summaries. The main limitation of this first version is that it intentionally rejects parallel tool-call replay and does not yet have a documentation page.

## Context and Orientation

The main user entry point for extra commands is `src/minisweagent/run/utilities/mini_extra.py`. SWE-bench environment construction lives in `src/minisweagent/run/benchmarks/swebench.py`, specifically `get_sb_environment`, which turns a resolved config and dataset instance into an execution environment. The normal coding-agent control flow lives in `src/minisweagent/agents/default.py`: `query()` samples the next assistant message, `execute_actions()` runs the parsed tool commands, and `step()` advances one executed step.

The merged verifier dataset row is a JSON object with at least `instance_id`, `step_index`, `message_index`, `history_trajectory`, and `actions`. In this repository, `history_trajectory` is already in the same tool-calling message format that current models use: assistant messages can have `tool_calls`, and corresponding tool outputs are separate `tool` messages. Each action entry contains a candidate command, label, gold flag, and often a `model_response` whose content serves as the candidate thought text.

The Monte Carlo implementation consists of two layers. The reusable layer is `src/minisweagent/run/extra/utils/trajectory_replay.py`, which extracts the task from the logged history, replays the live prefix into a fresh agent/environment pair, and synthesizes a forced branch assistant message from a candidate action. The user-facing layer is `src/minisweagent/run/extra/monte_carlo.py`, which loads rows, resolves SWE-bench instances, runs one rollout task per `(row, action, sample)` branch, saves trajectories, and writes summary files.

## Plan of Work

The implementation first loads and filters the merged rows, then resolves the SWE-bench instance lookup using the same dataset-loading path as the batch/single benchmark runners. For each rollout task, it resolves the config exactly the way SWE-bench does, applies optional model/environment overrides, builds a fresh agent and environment, and replays the row’s `history_trajectory` into that agent.

Replay works by iterating through the logged history, copying `system` and `user` messages directly, ignoring recorded `tool` messages, and for each `assistant` tool-call message: appending the assistant message, executing the extracted command through `agent.execute_actions(...)`, and keeping the live observation that comes back. This preserves the logged command sequence while allowing the environment outputs to diverge.

After the prefix is seeded, the runner synthesizes a branch assistant message from one candidate action. It preserves the candidate thought text from `model_response.content` when available, adds one bash tool call whose arguments exactly match the chosen candidate command, and stores the command in `extra.actions` so the normal agent execution path can use it. The forced branch is executed once. If that command submits immediately, the rollout ends there. Otherwise the runner continues by calling the normal `agent.step()` loop until the agent exits or the configured continuation step cap is reached.

Each rollout writes a `.traj.json` file under the output directory and records a compact JSON row with identifiers, candidate metadata, replay status, exit status, cost, and saved trajectory path. The top-level summary file aggregates counts for rows loaded, tasks planned/completed, replay failures, skipped-existing task counts, and terminal outcomes such as `Submitted`.

The runner also supports SWE-bench-like rerun controls. By default it skips rollout tasks already present in `results.jsonl`. `--redo-errors` reruns only tasks whose previous result had an error or missing trajectory output, while preserving successful existing rows. `--redo-existing` reruns all matching tasks and overrides `--redo-errors`.

When progress display is enabled, the runner uses the same `RunBatchProgressManager` pattern as the SWE-bench batch runner. Each rollout task gets its own live status line showing the instance, step, candidate/sample, and current phase such as replay, forced branch, or continuation step count.

After all rollout rows are written, the runner also writes two prediction files. `preds.json` contains one entry per rollout sample so downstream analysis can evaluate every sampled terminal patch. `preds_by_instance.json` preserves the old SWE-bench-style collapse by `instance_id`, choosing one rollout deterministically by preferring a non-empty `Submitted` patch, then any non-empty submission, then an empty patch if no rollout submitted successfully.

## Concrete Steps

Work from the repository root `/home/zkjzou/SWE-PRM/mini-swe-agent`.

Run the focused test suite for this feature:

    pytest -q tests/run/test_monte_carlo_rollout.py

Run a syntax check for the new modules:

    python -m py_compile src/minisweagent/run/extra/monte_carlo.py src/minisweagent/run/extra/utils/trajectory_replay.py tests/run/test_monte_carlo_rollout.py

Example invocation against merged verifier rows:

    mini-extra monte-carlo-rollout /path/to/merged_grouped_latest.jsonl \
      -c swebench.yaml \
      --subset verified \
      --split dev \
      --samples-per-action 3 \
      --max-rollout-steps 20 \
      --output-dir /tmp/mc_rollouts

Expected outcomes:
- `results.jsonl` appears in the output directory
- `summary.json` appears in the output directory
- rollout trajectories appear under `<output-dir>/<instance_id>/step_<step>/<candidate>__sample_<n>.traj.json`

## Validation and Acceptance

Validation for this implementation is:

    pytest -q tests/run/test_monte_carlo_rollout.py

The acceptance criteria are:
- the runner replays a logged prefix and does not reuse stale recorded tool outputs
- the forced candidate branch message contains the chosen command and candidate metadata
- one rollout trajectory is saved for every `(candidate action, sample)` pair
- `results.jsonl` records per-branch outcome fields such as `rollout_exit_status`, `forced_command`, and `trajectory_path`
- the new command is reachable through `mini-extra monte-carlo-rollout`

On 2026-03-08 this validation passed with `5 passed` in `tests/run/test_monte_carlo_rollout.py`.

## Idempotence and Recovery

The rollout runner is additive. Re-running it against the same output directory will overwrite `results.jsonl` and `summary.json`, while trajectory files for the same branch/sample path will be replaced by the latest run. If a rollout task fails before trajectory save, the error is still recorded in `results.jsonl` so analysis can continue across other tasks. Environment cleanup is attempted in a `finally` block for every rollout task.

## Artifacts and Notes

Important produced files:
- `src/minisweagent/run/extra/monte_carlo.py`
- `src/minisweagent/run/extra/utils/trajectory_replay.py`
- `tests/run/test_monte_carlo_rollout.py`

Representative result-row fields:

    {
      "instance_id": "repo__issue-1",
      "step_index": 1,
      "action_label": "gold",
      "sample_index": 0,
      "forced_command": "printf gold > branch.txt",
      "forced_thought": "Gold thought",
      "replay_status": "ok",
      "rollout_exit_status": "Submitted",
      "trajectory_path": ".../repo__issue-1/step_0001/gold__sample_000.traj.json"
    }

## Interfaces and Dependencies

The main callable introduced by this work is `minisweagent.run.extra.monte_carlo.generate_monte_carlo_rollouts(...)`. It depends on:
- `datasets.load_dataset` to resolve SWE-bench instances
- `minisweagent.run.benchmarks.swebench.get_sb_environment` for environment creation
- `minisweagent.models.get_model` for actor model creation
- `minisweagent.agents.get_agent` for constructing the rollout agent
- `minisweagent.utils.verifier_action_sampling.normalize_docent_message_for_model` and `extract_actions_from_assistant_message` for adapting merged-row messages into current model/runtime format

The replay helper introduced by this work is `minisweagent.run.extra.utils.trajectory_replay.seed_agent_from_history(agent, row)`, which mutates a fresh agent into the live state represented by the row’s prefix and returns a small replay summary.

Update (2026-03-08): Replaced the older saved-trajectory Monte Carlo design with the merged verifier-row implementation requested by the user, and recorded the completed implementation and test results.


Update (2026-03-08): Added automatic prediction export from Monte Carlo results so downstream evaluation can consume rollout outputs directly.

Update (2026-03-08): Added `--redo-existing` and `--redo-errors` handling for Monte Carlo tasks, using existing `results.jsonl` as the skip/rerun source of truth.

Update (2026-03-08): Added SWE-bench-style live progress output for Monte Carlo rollout tasks using `RunBatchProgressManager` and `rich.Live`.

Update (2026-03-09): `preds.json` now exports one patch per sampled rollout, while `preds_by_instance.json` preserves the single-patch-per-instance view. This matches Monte Carlo evaluation needs where all sampled terminal patches must remain visible.

Update (2026-03-09): Added `--row-start` and `--row-end` so Monte Carlo runs can target an exact 1-based inclusive row slice after `--instance` and `--step-index` filtering.

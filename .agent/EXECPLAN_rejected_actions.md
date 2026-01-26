# Rejected Action Sampling for Verifier Training

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds.

Follow /home/zkjzou/SWE-PRM/mini-swe-agent/.agent/PLANS.md. This document must remain fully self-contained and updated as the work advances.

## Purpose / Big Picture

After this change, a user can generate multiple rejected actions per step, sourced from models other than the expert coding agent, in two modes: online (during trajectory collection) and offline (after trajectories exist). The user can verify it works by running a batch trajectory run with online sampling enabled and by running a new offline sampler against saved trajectories, then inspecting the produced rejected-action files and confirming they are present, have the expected count per step, and do not use the expert model.

## Progress

- [x] (2026-01-25 19:25Z) Create the branch and add the ExecPlan file.
- [x] (2026-01-25 19:25Z) Implement rejected-action sampling data model and storage format.
- [x] (2026-01-25 19:25Z) Add online sampling hook to the agent execution path.
- [x] (2026-01-25 19:25Z) Add offline sampling CLI and utilities.
- [ ] (2026-01-25 19:25Z) Add tests and validation steps.
- [ ] (2026-01-25 19:25Z) Run validations and capture evidence.

## Surprises & Discoveries

- Observation: pytest is not available in the environment, so tests could not be executed.
  Evidence: `/bin/bash: pytest: command not found`

## Decision Log

- Decision: Store rejected actions in a per-trajectory sidecar JSONL file rather than embedding in the trajectory JSON.
  Rationale: Rejected actions can be large and numerous; a sidecar file keeps trajectories lightweight and allows independent re-generation in offline mode.
  Date/Author: 2026-01-24 / assistant

- Decision: Define an error-free expert action as the action parsed from the assistant message at each step and use the prompt messages immediately before that assistant message as the sampling input for rejected actions.
  Rationale: This mirrors the expert model's input and avoids leaking the expert action into the rejected proposals.
  Date/Author: 2026-01-24 / assistant

- Decision: Enforce that rejected actions are generated only by models in an explicit model pool that excludes the expert model by name and by class path.
  Rationale: The requirement is to sample from models other than the expert coding agent; explicit exclusion prevents accidental reuse.
  Date/Author: 2026-01-24 / assistant

- Decision: Online sampling appends a run identifier to the rejected-action filename, while offline sampling refuses to overwrite unless explicitly requested.
  Rationale: Online sampling should avoid cross-run collisions while still appending multiple steps; offline sampling must be safe and deterministic when re-run.
  Date/Author: 2026-01-25 / assistant

## Outcomes & Retrospective

No outcomes yet. Update after milestones complete.

## Context and Orientation

The main package is in src/minisweagent. The default agent logic is in src/minisweagent/agents/default.py. The agent stores messages and calls the model via Model.query, then parses and executes an action. Trajectories are saved by src/minisweagent/run/utils/save.py, which writes a JSON file with info, messages, and trajectory_format. Batch SWE-bench runs are implemented in src/minisweagent/run/extra/swebench.py. The extra CLI entry points are assembled in src/minisweagent/run/mini_extra.py. Configuration for batch swebench lives in src/minisweagent/config/extra/swebench.yaml. The tests for swebench are in tests/run/test_swebench.py.

Terminology used here:

A rejected action is a candidate action proposed by a non-expert model for the same step where the expert model produced the accepted action. The expert action is the action that the current agent actually executes. Online sampling means generating rejected actions during the run; offline sampling means generating them later by replaying prompts from saved trajectories.

## Plan of Work

Add a rejected action sampling subsystem that can be turned on or off. It must support an explicit pool of non-expert models and produce multiple rejected actions per step. The implementation uses a utility module that can build prompt messages, call alternate models, and persist samples to a sidecar JSONL file. This utility is used by the online execution path and by a new offline CLI command.

Update the configuration schema to include a rejected_action_sampling section with fields that describe mode, model pool, number of samples per step, and selection policy. Document and enforce that the expert model cannot appear in the pool.

For online sampling, add a hook in the default agent step loop. Capture the prompt messages immediately before the expert model query. After the expert action is produced, call the rejected-action sampler with the same prompt messages to produce k alternative actions. These must not be executed. Persist them in a sidecar file adjacent to the trajectory, with a naming convention like <trajectory_stem>.rejected.<run_id>.jsonl. Each JSONL line should include the step index, expert action, rejected action, rejected model ID, and metadata like timestamp and mode.

For offline sampling, implement a new CLI under src/minisweagent/run/extra that reads existing trajectory JSON files, reconstructs the prompt messages for each step, and generates rejected actions using the same sampler utility. It must support batching and allow output to the same sidecar naming convention. It must not modify the original trajectory file.

Add tests that validate that online sampling records the right count and does not use the expert model, and that offline sampling produces the same schema. Provide an acceptance test that runs the dummy swebench dataset with online sampling enabled and verifies the sidecar file content.

## Concrete Steps

1) Create the branch and confirm clean working directory.
   Working directory: /home/zkjzou/SWE-PRM/mini-swe-agent
   Command:
     git checkout -b verifier-rejected-actions

2) Add the rejected action schema and storage utilities.
   Edit or add:
   - src/minisweagent/run/utils/rejected_actions.py
   This module should define a sampler class and JSONL writer for rejected actions.

3) Add config and wiring for online sampling.
   Edit:
   - src/minisweagent/config/extra/swebench.yaml
   - src/minisweagent/agents/default.py
   Ensure the expert model is excluded from the pool by name and class, and ensure the sampler is only invoked when enabled.

4) Add offline sampling CLI.
   Add:
   - src/minisweagent/run/extra/rejected_actions.py
   Register in:
   - src/minisweagent/run/mini_extra.py
   The CLI should accept a path (file or directory), a model pool config, and output directory or overwrite flag.

5) Add tests.
   Add or update:
   - tests/run/test_swebench.py
   - tests/run/test_rejected_actions.py (new)
   Include fixtures that simulate a short trajectory and confirm the JSONL schema and model exclusions.

6) Run validation commands.
   Working directory: /home/zkjzou/SWE-PRM/mini-swe-agent
   Commands:
     pytest -n auto tests/run/test_rejected_actions.py
     pytest -n auto tests/run/test_swebench.py -k rejected

Expected evidence snippets:
  - The new tests pass.
  - A sample run creates <instance_id>.rejected.<run_id>.jsonl with k entries per step and rejected_model_id never equals the expert model.

## Validation and Acceptance

Acceptance is met when a user can:

- Run a batch swebench job with online sampling enabled and observe a sidecar JSONL file in each instance directory. The sidecar must include a line per rejected action with step index, expert action, rejected action, rejected model ID, and mode set to online.
- Run the offline sampler against existing trajectories and see the sidecar created or updated with mode set to offline, without altering the trajectory JSON.
- Confirm that no rejected action is generated by the expert model by checking the rejected_model_id fields.

Concrete manual validation example:

- Run the dummy swebench dataset with online sampling enabled and workers=1 to simplify inspection.
- Inspect one trajectory directory and verify that the rejected JSONL file has k lines per step.
- Run the offline sampler against the same trajectory and verify it either overwrites or appends based on the chosen flag.

## Idempotence and Recovery

The offline sampler must be safe to re-run. Provide a clear flag (for example, --overwrite) that replaces an existing sidecar file. If the flag is not set, the command should refuse to overwrite and exit with a clear error message. Online sampling should append multiple steps to a single run-specific sidecar file; repeated runs should create a new file name that includes a run identifier. If a run is interrupted, the user can delete the sidecar file and re-run the sampler without affecting the trajectory file.

## Artifacts and Notes

Example JSONL entry format (one line per rejected action, shown as indented text):
  {"step_index": 3, "expert_action": "ls -la", "rejected_action": "git status", "rejected_model_id": "alt-model-1", "mode": "online", "timestamp": "2026-01-24T00:00:00Z"}

Example CLI usage (shown as indented text):
  mini-extra rejected-actions /path/to/trajectories --config /path/to/swebench.yaml --overwrite

## Interfaces and Dependencies

The rejected action sampler utility lives in src/minisweagent/run/utils/rejected_actions.py and exposes:

- RejectedActionSampler with methods sample_records(...) to return records and sample_and_write(...) to append to the configured JSONL file.
- write_rejected_actions(path, records, overwrite) for direct JSONL writes.
- extract_action_from_response(content, action_regex) to parse expert actions consistently.

The online sampling hook is added to the agent run loop in src/minisweagent/agents/default.py. It captures prompt messages before the expert query, executes the expert query as usual, then calls the sampler with the same prompt messages and the action regex from the agent config. The offline CLI reuses the sampler and parses the expert action using the action regex stored in the trajectory info config (info.config.agent.action_regex), falling back to the default regex if not present.

Plan Update (2026-01-25): Updated progress to reflect completed implementation steps and aligned interfaces and filenames with the current code changes, including run_id-based filenames and overwrite behavior. Recorded missing pytest availability during validation.

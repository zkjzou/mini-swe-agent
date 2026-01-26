# Add a Configurable Verifier That Selects the Best Action From Sampled Candidates

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan must be maintained in accordance with `.agent/PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the agent can sample multiple candidate actions per step, send those candidates to a configurable verifier, and then execute the verifier’s chosen action. Users can switch verifier types and prompts purely through configuration, and they can run the verifier on a different model than the main coding agent. The behavior is visible by running the agent with a config that enables the verifier and observing that the executed action corresponds to the verifier’s selection rather than the first sampled candidate. The candidate actions and verifier output are also recorded in the saved trajectory JSON in the same overall structure as existing `.traj.json` files (i.e., under message `extra` fields).

## Progress

- [x] (2026-01-25 01:10Z) Initialize work on branch `feature-verifier-rejected-actions`.
- [x] (2026-01-25 01:12Z) Implemented candidate sampling and verifier selection in `DefaultAgent`.
- [x] (2026-01-25 01:14Z) Added verifier types (`first_valid`, `llm`) and configuration schema.
- [x] (2026-01-25 01:16Z) Updated configs/docs and added tests validating selection behavior and trajectory metadata.

## Surprises & Discoveries

- Observation: All model implementations return the raw response in `extra.response`, which can contain multiple choices if `n>1` is used.
  Evidence: `src/minisweagent/models/litellm_model.py`, `openrouter_model.py`, `portkey_model.py` return `extra.response` with `choices`.

## Decision Log

- Decision: Implement verifier integration at the agent layer (not as a model wrapper), so only the selected response is added to conversation history.
  Rationale: Keeps message history consistent and avoids altering all model classes or `get_model` behavior.
  Date/Author: 2026-01-24 / Codex
- Decision: Support at least two verifier types initially: `llm` (model-based) and `first_valid` (rule-based).
  Rationale: Enables experimentation with different verifier types and gives a deterministic baseline for tests.
  Date/Author: 2026-01-24 / Codex
- Decision: Provide optional efficient sampling using `n` when configured, with a safe sequential fallback.
  Rationale: Meets the “efficient sampling” requirement without breaking models that do not support `n`.
  Date/Author: 2026-01-24 / Codex
- Decision: Store candidate actions and verifier output in the selected assistant message’s `extra` field so `save_traj` captures it without changing the trajectory schema.
  Rationale: Existing trajectories already store model metadata under `extra`, so this is backwards-compatible and requires no changes to `save_traj`.
  Date/Author: 2026-01-24 / Codex

## Outcomes & Retrospective

Not started yet.

## Context and Orientation

The core agent lives in `src/minisweagent/agents/default.py`. `DefaultAgent.step()` currently calls `DefaultAgent.query()` once per step, which both queries the model and appends the assistant message to `self.messages`. It then parses a single action and executes it in the environment. Interactive variants in `src/minisweagent/agents/interactive.py` and `src/minisweagent/agents/interactive_textual.py` override `query` to support human-in-the-loop modes but still rely on `DefaultAgent.step()`.

Models are abstracted via the `Model` protocol in `src/minisweagent/__init__.py` and implemented in `src/minisweagent/models/`. Most models return a dict with `content` and `extra.response`. Configs are YAML files under `src/minisweagent/config/`, with `agent`, `model`, and `environment` sections. Tests live under `tests/`, with agent tests in `tests/agents/` and CLI tests in `tests/run/`.

A “verifier” in this plan is a component that receives multiple candidate responses (each containing a proposed action), judges them, and selects the best candidate to execute. The verifier may use a different model than the main agent.

## Plan of Work

First, create a dedicated feature branch so the implementation is isolated. Then extend the agent’s control flow to support candidate sampling and verifier selection without breaking existing single-sample behavior. The main refactor is to separate “raw model calls” from “adding the selected assistant message to history,” so we can sample multiple candidates while keeping only the chosen response in `self.messages`.

Next, add a small verifier module under `src/minisweagent/verifiers/` with a base interface and two implementations: a simple rule-based verifier (`first_valid`) and an LLM-based verifier (`llm`). The LLM verifier will accept a separate `Model` instance and render a configurable prompt that includes candidate actions (and optionally raw candidate text). It will parse the verifier’s response with a configurable regex, defaulting to a 1-based choice index.

Then extend `AgentConfig` to include a structured verifier config and candidate sampling config. This keeps experimentation easy by editing YAML rather than code. The agent should instantiate the verifier based on config and pass its decisions along as metadata, storing selection details in the chosen assistant message’s `extra` field (to be saved in the trajectory).

Finally, update the default configs to include a disabled verifier block with example prompt templates, add tests that verify selection and fallback behavior, and update docs to mention the new verifier step in the control flow.

## Concrete Steps

All commands should run from `/home/zkjzou/SWE-PRM/mini-swe-agent`.

1) Create a separate branch for the feature.

    git switch -c feature/verifier-action-selection

    Expected output mentions the new branch name.

2) Add verifier configuration models and sampling config in `src/minisweagent/agents/default.py` (or a new module under `src/minisweagent/verifiers/` if cleaner). The config must include:
- `enabled` flag.
- `type` or `verifier_class` to select verifier implementation.
- `model` configuration for the verifier model (separate from the agent model).
- Prompt templates (system + selection).
- `selection_regex` and `selection_index_base` (1-based by default).
- Sampling settings: `num_candidates`, `use_n`, optional `sampling_kwargs` for temperature/top_p/max_tokens.
- Fallback behavior when all candidates are invalid or selection parsing fails.

3) Refactor `DefaultAgent.query()` to:
- Check limits once per step.
- Sample `num_candidates` candidate responses without appending them to `self.messages`.
- Invoke the verifier if enabled, otherwise choose the first candidate.
- Append only the selected candidate to `self.messages`, with `extra` metadata including candidate list, selection index, and verifier info so it appears in the saved trajectory JSON.
- Return the selected response for the existing `get_observation()` flow.

4) Implement candidate sampling:
- Add a helper method (e.g., `_query_once`) that calls `self.model.query()` without touching message history.
- If `use_n` is true and `num_candidates > 1`, call the model once with `n=num_candidates` and parse `extra.response.choices` into multiple candidate responses.
- Otherwise call `_query_once` multiple times to gather candidates.
- Ensure candidate parsing does not mutate `self.messages`.

5) Create verifier module(s):
- `src/minisweagent/verifiers/base.py` defining a minimal interface (e.g., a class with `select(candidates, messages, task)`).
- `src/minisweagent/verifiers/first_valid.py` that picks the first candidate with a parseable action (fallback to index 0).
- `src/minisweagent/verifiers/llm.py` that:
  - Renders prompts from config templates and candidate data.
  - Calls the verifier model.
  - Parses selection via regex and maps to a candidate index.
  - Provides a safe fallback if parsing fails.

6) Wire verifier creation into the agent:
- In `DefaultAgent.__init__`, create the verifier instance based on config.
- Instantiate the verifier model using `get_model` with the verifier config (this is what enables a different model for the verifier).
- Keep verifier model state separate from `AgentConfig` so `save_traj` remains JSON-serializable.

7) Update config files:
- Add a `verifier` block to `src/minisweagent/config/default.yaml` and `src/minisweagent/config/mini.yaml` with `enabled: false` by default and a complete prompt template example.
- Consider adding to `github_issue.yaml` and relevant `extra/` configs if they are commonly used.

8) Update tests:
- Add new tests under `tests/agents/` to verify that:
  - With verifier enabled, the selected action corresponds to the verifier choice.
  - With `first_valid`, the first parseable candidate is chosen.
  - When all candidates are invalid, behavior falls back to the configured default (e.g., first candidate, then `FormatError`).
- Extend or add a stub model (or test helper) that simulates multi-choice responses for `use_n`.
- Update any tests that rely on `DefaultAgent.query()` signature or message counts, if needed.

9) Update docs:
- In `docs/advanced/control_flow.md`, mention the optional verifier step between “query model” and “parse action.”
- Add a brief note to `docs/reference/agents/default.md` about the verifier config and sampling.

## Validation and Acceptance

- Run unit tests: `pytest -n auto`.
- Add at least one new test that fails before the change and passes after, demonstrating that the verifier’s selection is what gets executed.
- Add a test (or assertion in an existing test) that the saved trajectory JSON includes the verifier metadata in the selected assistant message’s `extra` field.
- Manual sanity check: run `mini` with a config that enables the verifier and sets `num_candidates > 1`, and confirm the action executed matches the verifier’s selected index.

## Idempotence and Recovery

- All edits are additive or localized refactors; re-running the steps should be safe.
- If `use_n` causes model errors, set `use_n: false` and re-run tests to confirm the fallback path works.
- If verifier parsing fails, the fallback selection logic prevents crashes and preserves behavior.

## Artifacts and Notes

Keep any updated or new config snippets minimal, and document them in prose in the plan and in the config files themselves. Store verifier decisions in the selected assistant message’s `extra` metadata so `save_traj` captures enough context for debugging.

## Interfaces and Dependencies

- In `src/minisweagent/agents/default.py`, extend `AgentConfig` with a nested verifier config and sampling config. Keep types simple and JSON-serializable.
- Add a new verifier package in `src/minisweagent/verifiers/` with:
  - A base verifier interface.
  - `FirstValidVerifier` (rule-based).
  - `LLMVerifier` (uses a separate `Model` instance).
- Ensure `DefaultAgent` can:
  - Call a raw model query without appending to messages.
  - Sample multiple candidates.
  - Select a candidate via the verifier.
  - Append only the selected candidate to history.

The final behavior must be: for each agent step, multiple candidates are sampled; the verifier chooses one; only that one is executed and added to the conversation, with metadata captured for inspection.

Plan reset note (2026-01-25): Reset progress tracking to align with the new worktree branch request.

# Extend Verifier Action Evaluation To All Current Variants

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

Expand `verifier_action_evaluation` so offline evaluation can cover every verifier variant currently represented in
the repo, not just the coarse `llm` and `reward_model` buckets. The evaluator should support `first_valid`, all
registered prompt variants, checklist and dynamic-checklist prompt semantics, and per-variant reporting.

## Progress

- [x] (2026-03-10 00:00Z) Confirmed the current evaluator only supports `llm` and `reward_model`, while config and
      prompt assets define a broader set of verifier variants.
- [x] (2026-03-10 00:10Z) Added an evaluator-side verifier variant registry with per-variant config overrides.
- [x] (2026-03-10 00:20Z) Extended evaluation output and summaries to report concrete variants plus aggregate
      per-verifier metrics.
- [x] (2026-03-10 00:30Z) Updated the CLI and tests for variant-aware evaluation, world-prompt handling, checklist
      metadata, and `first_valid`.
- [x] (2026-03-10 00:40Z) Ran targeted syntax checks and pytest for the updated evaluator and CLI paths.

## Surprises & Discoveries

- Observation: The benchmark prompt profiles only inject `prompt_name`/`prompt_dir`; checklist prompt variants still
  need evaluator-side config overrides for `checklist_mode`, `checklist_dynamic`, and update behavior.
  Evidence: `swebench.yaml` prompt profiles list prompt aliases only, while checklist rendering in the evaluator is
  gated by `VerifierConfig.checklist_mode == "issue_progress"`.

## Decision Log

- Decision: Use an explicit evaluator-side registry for supported variants instead of inferring everything from
  prompt names.
  Rationale: Several variants need behavior overrides beyond prompt aliasing, especially checklist and dynamic
  checklist modes.
  Date/Author: 2026-03-10 / Codex
- Decision: Keep both `per_variant` and aggregate `per_verifier` metrics in the evaluation summary.
  Rationale: Variant-level visibility is needed for analysis, but aggregate base-type metrics preserve backward
  compatibility for existing consumers.
  Date/Author: 2026-03-10 / Codex

## Outcomes & Retrospective

- Result: `verifier_action_evaluation` now evaluates all current verifier variants through an explicit registry,
  emits per-variant plus aggregate summaries, supports `first_valid`, and exposes variant selection through the CLI.
- Validation:
  - `python -m py_compile src/minisweagent/utils/verifier_action_evaluation.py src/minisweagent/run/utilities/evaluate_verifier_actions.py tests/utils/test_verifier_action_evaluation.py tests/run/test_evaluate_verifier_actions_cli.py`
  - `pytest -q tests/utils/test_verifier_action_evaluation.py tests/run/test_evaluate_verifier_actions_cli.py`

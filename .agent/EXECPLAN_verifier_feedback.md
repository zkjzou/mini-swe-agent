# Add Delayed Verifier Feedback For Reward-Model Action Selection

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

Extend the existing `reward_model` verifier so it not only scores sampled candidate actions and picks the best one,
but also emits concise feedback for the selected action. That feedback should not trigger a same-step retry; instead,
the chosen action is executed normally and the coding model receives the prior step's selected verifier feedback on the
next actor query through a dedicated prompt template.

## Progress

- [x] (2026-03-08 00:00Z) Confirmed the current verifier architecture already supports candidate scoring and
      metadata capture, but does not feed critique back to the actor.
- [x] (2026-03-08 00:10Z) Implemented reward-verifier feedback parsing and selected-feedback metadata.
- [x] (2026-03-08 00:15Z) Injected prior-step verifier feedback into outbound actor context with a dedicated template.
- [x] (2026-03-08 00:20Z) Updated reward prompt templates, base configs, and reward-verifier tests.
- [x] (2026-03-08 00:25Z) Verified with targeted syntax checks and pytest slices.

## Surprises & Discoveries

- Observation: Actor `system_template` and `instance_template` are rendered once at `run()` startup, so next-step
  verifier feedback cannot be implemented by only editing those templates.
  Evidence: `DefaultAgent.run()` renders the initial two messages once, while `_query_once()` forwards `self.messages`
  directly on later steps.

## Decision Log

- Decision: Implement actor feedback as an ephemeral, configurable outbound message template rather than mutating
  stored history with raw verifier output.
  Rationale: This preserves trajectory history while still giving the actor an explicit prompt slot every query.
  Date/Author: 2026-03-08 / Codex
- Decision: Extend `reward_model` instead of adding a new verifier type.
  Rationale: The existing per-candidate scoring path already matches the required control flow and metadata shape.
  Date/Author: 2026-03-08 / Codex
- Decision: Surface only the previous step's selected action feedback to the actor.
  Rationale: Keeps prompt size bounded and ties the critique directly to the action that was actually executed.
  Date/Author: 2026-03-08 / Codex

## Outcomes & Retrospective

- Result: `reward_model` verifier now captures per-candidate critique, stores selected feedback in trajectory metadata,
  and passes the selected action's critique/score into the next actor query via a configurable outbound message.
- Validation:
  - `python -m py_compile src/minisweagent/agents/default.py src/minisweagent/verifiers/reward_model.py tests/agents/test_reward_verifier.py`
  - `pytest -q tests/agents/test_reward_verifier.py tests/agents/test_verifier.py tests/agents/test_default.py`

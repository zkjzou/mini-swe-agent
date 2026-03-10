# Add Multi-Turn Verifier History Passing

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

After this change, verifier prompts can receive prior trajectory context as an actual multi-turn chat transcript instead of only as a flattened `Recent steps ...` block inside the final user prompt. The new behavior is opt-in via verifier config, applies across selection, reward, and checklist verifier chats, and preserves the existing single-prompt behavior by default. Users can validate the feature by enabling `agent.verifier.history_message_format=multi_turn_chat` and observing that captured verifier inputs contain `system + replayed history turns + final user prompt`, with no injected verifier feedback replayed into history.

## Progress

- [x] (2026-03-10 18:55Z) Added verifier config and shared message-building helpers for single-prompt vs multi-turn verifier history.
- [x] (2026-03-10 19:05Z) Updated built-in verifier prompt variants to conditionally omit inline history in multi-turn mode.
- [x] (2026-03-10 19:12Z) Added focused tests for selection, reward, checklist, agent integration, and offline evaluation paths.
- [x] (2026-03-10 19:24Z) Extended multi-turn replay to include assistant tool-call metadata and tool output messages.
- [x] (2026-03-10 19:39Z) Preserved assistant tool calls in structured chat format (`tool_calls`) instead of flattening them into content text.
- [x] (2026-03-10 19:50Z) Auto-enabled checklist mode when checklist prompt variants are selected directly by `prompt_name`.

## Surprises & Discoveries

- Observation: verifier history in the live agent path is sourced from executed assistant/observation steps, so calling `query()` twice without `execute_actions()` only yields assistant history and no observation turn.
  Evidence: `DefaultAgent.query()` builds verifier history from `self.messages`, and tests that call `query()` directly produced `["system", "assistant", "user"]` verifier inputs in multi-turn mode.
- Observation: the offline verifier evaluation fixture replays only assistant content from its sample `history_trajectory`, because the fixture uses a tool message rather than a user observation.
  Evidence: `tests/utils/test_verifier_action_evaluation.py::_make_row()` plus captured verifier inputs during multi-turn evaluation.

## Decision Log

- Decision: make multi-turn verifier history opt-in with `history_message_format` instead of replacing the current prompt shape.
  Rationale: preserves existing prompt compatibility and avoids silently changing current verifier behavior.
  Date/Author: 2026-03-10 / Codex
- Decision: replay original `user`/`assistant` turns and append a final `user` verifier instruction, rather than synthesizing `Step N` wrappers.
  Rationale: keeps the verifier transcript closest to the original trajectory and works across model backends that already accept chat message arrays.
  Date/Author: 2026-03-10 / Codex
- Decision: include tool-call summaries on assistant turns and replay tool outputs as `tool` messages in multi-turn history.
  Rationale: verifier decisions often depend on the exact command invocation and resulting tool output, not just free-form assistant text.
  Date/Author: 2026-03-10 / Codex
- Decision: preserve assistant tool calls as structured `tool_calls` fields when available, matching the original chat transcript shape.
  Rationale: this keeps replayed history closer to the source trajectory and matches downstream expectations for tool-using assistant messages.
  Date/Author: 2026-03-10 / Codex
- Decision: infer checklist mode/settings from checklist prompt names (`checklist/*`, `checklist_v2/*`, `dynamic_checklist_*/*`) even when scripts set only `prompt_name`.
  Rationale: checklist prompt variants require `checklist_text`, so direct prompt selection should not require redundant manual checklist-mode config to avoid runtime failures.
  Date/Author: 2026-03-10 / Codex
- Decision: exclude verifier feedback and verifier metadata from replayed history.
  Rationale: the verifier should judge the actor trajectory itself, not its own previous outputs or injected critique text.
  Date/Author: 2026-03-10 / Codex

## Outcomes & Retrospective

Implemented with focused verification:

- `pytest -q tests/verifiers/test_query_path.py tests/verifiers/test_checklist.py tests/agents/test_reward_verifier.py tests/utils/test_verifier_action_evaluation.py`
- Result: `34 passed`
- Follow-up verification after adding tool-call/tool-output replay:
  `pytest -q tests/verifiers/test_query_path.py tests/utils/test_verifier_action_evaluation.py tests/agents/test_verifier.py tests/agents/test_reward_verifier.py tests/verifiers/test_checklist.py`
  Result: `57 passed`

The shared message-builder path now keeps selection, reward, checklist generation, and offline evaluation behavior aligned.

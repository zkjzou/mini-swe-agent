from __future__ import annotations

from types import SimpleNamespace

from minisweagent.verifiers.checklist import (
    dedupe_checklist_items,
    generate_issue_checklist,
    infer_checklist_prompt_settings,
    load_checklist_generator_templates,
    normalize_checklist_items,
    parse_checklist_items,
    parse_checklist_rubric,
    resolve_checklist_generator_prompt_name,
    resolve_checklist_output_format,
    sanitize_checklist_generation_output,
)


def test_parse_checklist_items_from_numbered_lines():
    content = "CHECKLIST:\n1. Reproduce bug\n2) Implement fix\n- Run targeted tests\n"
    items = parse_checklist_items(content, item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")
    assert items == ["Reproduce bug", "Implement fix", "Run targeted tests"]


def test_normalize_checklist_items_dedupes_and_fills_min_items():
    items = [
        "Reproduce bug",
        "Reproduce   bug",
        "Implement fix",
    ]
    normalized = normalize_checklist_items(items, min_items=4, max_items=5)
    assert normalized[0] == "Reproduce bug"
    assert normalized[1] == "Implement fix"
    assert len(normalized) == 4


def test_dedupe_checklist_items_preserves_order():
    items = ["Reproduce bug", "Implement fix", "Reproduce   bug", "Run tests"]
    deduped = dedupe_checklist_items(items)
    assert deduped == ["Reproduce bug", "Implement fix", "Run tests"]


def test_parse_checklist_rubric_yaml():
    content = """
rubric:
  - id: S1
    stage: understanding
    weight: 3
    description: Identifies the root cause in src/foo.py
    observable_signal: Mentions failing stack trace in logs
  - id: S2
    phase: verification
    weight: 2
    description: Runs targeted tests for astropy__astropy-7336
    done_when: pytest output reports passing tests
"""
    parsed = parse_checklist_rubric(content)
    assert parsed == [
        {
            "id": "S1",
            "stage": "understanding",
            "weight": 3,
            "description": "Identifies the root cause in src/foo.py",
            "observable_signal": "Mentions failing stack trace in logs",
        },
        {
            "id": "S2",
            "stage": "verification",
            "weight": 2,
            "description": "Runs targeted tests for astropy__astropy-7336",
            "observable_signal": "pytest output reports passing tests",
        },
    ]


def test_resolve_checklist_output_format_autodetects_checklist_v2():
    config = SimpleNamespace(prompt_name="checklist_v2/verifier", checklist_output_format="auto")
    assert resolve_checklist_output_format(config) == "rubric_yaml"


def test_resolve_checklist_output_format_autodetects_ultimate_v2():
    config = SimpleNamespace(prompt_name="ultimate_v2/verifier", checklist_output_format="auto")
    assert resolve_checklist_output_format(config) == "rubric_yaml"


def test_resolve_checklist_output_format_autodetects_dynamic_ultimate_v2():
    config = SimpleNamespace(
        prompt_name="ultimate_v2_dynamic_checklist_regenerate/verifier",
        checklist_output_format="auto",
    )
    assert resolve_checklist_output_format(config) == "rubric_yaml"


def test_resolve_checklist_output_format_autodetects_ultimate_v2_mini():
    config = SimpleNamespace(prompt_name="ultimate_v2_mini/verifier", checklist_output_format="auto")
    assert resolve_checklist_output_format(config) == "rubric_yaml"


def test_load_checklist_generator_templates_supports_static_success_v2():
    config = SimpleNamespace(checklist_generator_prompt_name="static_success_v2")
    system_template, prompt_template = load_checklist_generator_templates(config)

    assert system_template is not None
    assert prompt_template is not None
    assert "verifier-facing process rubric" in prompt_template


def test_infer_checklist_prompt_settings_enables_ultimate_v2_variants():
    assert infer_checklist_prompt_settings("ultimate_v2/verifier") == {
        "checklist_mode": "issue_progress",
        "checklist_dynamic": False,
        "checklist_update_mode": "regenerate",
    }
    assert infer_checklist_prompt_settings("ultimate_v2_mini/reward") == {
        "checklist_mode": "issue_progress",
        "checklist_dynamic": False,
        "checklist_update_mode": "regenerate",
    }


def test_generate_issue_checklist_uses_model_query_and_parses_items():
    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="system",
        checklist_prompt_template="task: {{ task }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
    )
    output = generate_issue_checklist(_QueryOnlyModel(), config, template_vars={"task": "sample issue", "messages": []})

    assert output["items"] == ["Reproduce issue", "Patch source code", "Run tests"]
    assert output["response_cost"] == 0.25
    assert "CHECKLIST" in output["raw_output"]


def test_generate_issue_checklist_parses_rubric_without_min_max():
    class _RubricModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: S1\n"
                    "    phase: localize\n"
                    "    weight: 3\n"
                    "    description: Locates failure in src/minisweagent/agents/default.py\n"
                    "    done_when: Stack trace points to _prepare_checklist_template_vars\n"
                    "  - id: S2\n"
                    "    stage: verification\n"
                    "    weight: 2\n"
                    "    description: Verifies checklist scores are parsed for each candidate\n"
                    "    observable_signal: Trajectory stores non-empty checklist item scores\n"
                ),
                "extra": {"cost": 0.12},
            }

    config = SimpleNamespace(
        prompt_name="checklist_v2/verifier",
        checklist_output_format="auto",
        checklist_system_template="system",
        checklist_prompt_template="task: {{ task }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
    )
    output = generate_issue_checklist(_RubricModel(), config, template_vars={"task": "sample issue", "messages": []})

    assert output["checklist_output_format"] == "rubric_yaml"
    assert output["items"] == [
        "Locates failure in src/minisweagent/agents/default.py",
        "Verifies checklist scores are parsed for each candidate",
    ]
    assert output["rubric_items"][0]["id"] == "S1"
    assert output["response_cost"] == 0.12


def test_generate_issue_checklist_can_include_rendered_inputs():
    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="system {{ task }}",
        checklist_prompt_template="task: {{ task }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
        include_inputs_in_output=True,
    )
    output = generate_issue_checklist(_QueryOnlyModel(), config, template_vars={"task": "sample issue", "messages": []})

    assert output["input"]["messages"] == [
        {"role": "system", "content": "system sample issue"},
        {"role": "user", "content": "task: sample issue"},
    ]


def test_generate_issue_checklist_strips_think_blocks_from_rendered_inputs():
    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="system <think>hidden</think> {{ task }}",
        checklist_prompt_template="Issue description: {{ task }}\nRecent: {{ messages[0].content }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
        include_inputs_in_output=True,
        history_message_format="multi_turn_chat",
    )
    output = generate_issue_checklist(
        _QueryOnlyModel(),
        config,
        template_vars={
            "task": "sample issue",
            "messages": [
                {"role": "assistant", "content": "<think>internal reasoning</think>\n\nInspect parser"},
                {"role": "user", "content": "Found failing test output"},
            ],
        },
    )

    assert "sample issue" in output["input"]["messages"][0]["content"]
    assert "<think>" not in output["input"]["messages"][0]["content"]
    assert "hidden" not in output["input"]["messages"][0]["content"]
    assert "Inspect parser" in output["input"]["messages"][-1]["content"]
    assert "internal reasoning" not in output["input"]["messages"][-1]["content"]
    assert "<think>" not in output["input"]["messages"][-1]["content"]


def test_generate_issue_checklist_can_replay_history_as_multi_turn_chat():
    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="system",
        checklist_prompt_template="Issue description: {{ task }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
        include_inputs_in_output=True,
        history_message_format="multi_turn_chat",
    )
    output = generate_issue_checklist(
        _QueryOnlyModel(),
        config,
        template_vars={
            "task": "sample issue",
            "messages": [
                {"role": "assistant", "content": "Inspect parser"},
                {"role": "user", "content": "Found failing test output"},
            ],
        },
    )

    assert output["input"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "Inspect parser"},
        {"role": "user", "content": "Found failing test output"},
        {"role": "user", "content": "Issue description: sample issue"},
    ]


def test_load_checklist_generator_templates_reads_prompt_family(tmp_path):
    prompt_dir = tmp_path / "prompts" / "checklist_generator" / "dynamic_success"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.jinja").write_text("generator system")
    (prompt_dir / "prompt.jinja").write_text("generator prompt")

    config = SimpleNamespace(
        checklist_generator_prompt_name="dynamic_success",
        checklist_generator_prompt_dir=str(tmp_path / "prompts" / "checklist_generator"),
    )

    system_template, prompt_template = load_checklist_generator_templates(config)

    assert system_template == "generator system"
    assert prompt_template == "generator prompt"


def test_resolve_checklist_generator_prompt_name_defaults_from_mode():
    config = SimpleNamespace(checklist_generator_mode="trajectory_dynamic", checklist_generator_prompt_name=None)

    assert resolve_checklist_generator_prompt_name(config) == "dynamic_success"


def test_generate_issue_checklist_uses_checklist_generator_prompt_override(tmp_path):
    prompt_dir = tmp_path / "prompts" / "checklist_generator" / "trajectory_success"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.jinja").write_text("system {{ task }}")
    (prompt_dir / "prompt.jinja").write_text("trajectory {{ trajectory_text }}")

    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="fallback system",
        checklist_prompt_template="fallback prompt",
        checklist_generator_prompt_name="trajectory_success",
        checklist_generator_prompt_dir=str(tmp_path / "prompts" / "checklist_generator"),
        checklist_generator_mode="trajectory_success",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
        include_inputs_in_output=True,
    )

    output = generate_issue_checklist(
        _QueryOnlyModel(),
        config,
        template_vars={"task": "sample issue", "messages": [], "trajectory_text": "assistant: inspect file"},
    )

    assert output["input"]["messages"][0]["content"] == "system sample issue"
    assert output["input"]["messages"][-1]["content"] == "trajectory assistant: inspect file"


def test_sanitize_checklist_generation_output_strips_future_only_terms():
    config = SimpleNamespace(
        checklist_generator_mode="trajectory_dynamic",
        checklist_generator_validate_grounding=True,
        checklist_output_format="rubric_yaml",
        checklist_min_items=3,
        checklist_max_items=5,
    )

    sanitized = sanitize_checklist_generation_output(
        config,
        template_vars={
            "task": "Fix bug",
            "messages": [{"role": "assistant", "content": "Inspect parser.py"}],
            "future_steps": ["assistant: edit hidden_future.py to update FutureSymbol"],
            "future_steps_text": "1. assistant: edit hidden_future.py to update FutureSymbol",
        },
        items=["Inspect hidden_future.py and update FutureSymbol"],
        rubric_items=[
            {
                "id": "S1",
                "description": "Inspect hidden_future.py and update FutureSymbol",
                "stage": "fix",
                "observable_signal": "FutureSymbol is updated in hidden_future.py",
            }
        ],
    )

    assert sanitized["guardrail"]["mode"] == "trajectory_dynamic"
    assert "hidden_future.py" in sanitized["guardrail"]["sanitized_terms"]
    assert sanitized["items"][0] == "Inspect relevant implementation detail and update relevant code element"
    assert sanitized["rubric_items"][0]["description"] == sanitized["items"][0]


def test_generate_issue_checklist_dynamic_mode_uses_inferred_prompt_and_full_future_steps():
    class _DynamicRubricModel:
        def query(self, messages, **kwargs):
            self.messages = messages
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: D1\n"
                    "    phase: validate\n"
                    "    weight: 3\n"
                    "    description: Validate src/future_only.py after the change\n"
                    "    done_when: src/future_only.py passes focused checks\n"
                ),
                "extra": {"cost": 0.3},
            }

    config = SimpleNamespace(
        checklist_generator_mode="trajectory_dynamic",
        checklist_generator_prompt_name=None,
        checklist_generator_prompt_dir="prompts/checklist_generator",
        checklist_generator_validate_grounding=True,
        checklist_output_format="rubric_yaml",
        checklist_system_template="fallback system",
        checklist_prompt_template="fallback prompt",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
        include_inputs_in_output=True,
        history_message_format="single_prompt",
    )

    output = generate_issue_checklist(
        _DynamicRubricModel(),
        config,
        template_vars={
            "task": "Fix validation flow",
            "steps": [
                [
                    {"role": "assistant", "content": "Inspect current validation logic"},
                    {"role": "user", "content": "Observed problem in src/current.py"},
                ]
            ],
            "all_steps": [
                [
                    {"role": "assistant", "content": "Inspect current validation logic"},
                    {"role": "user", "content": "Observed problem in src/current.py"},
                ],
                [
                    {"role": "assistant", "content": "Edit src/future_only.py"},
                    {"role": "user", "content": "Focused test now passes"},
                ],
                [
                    {"role": "assistant", "content": "Run regression suite"},
                    {"role": "user", "content": "No regressions found"},
                ],
            ],
        },
    )

    rendered_prompt = output["input"]["messages"][-1]["content"]
    assert "Full future steps after the current step" in rendered_prompt
    assert "Edit src/future_only.py" in rendered_prompt
    assert output["generator_prompt_name"] == "dynamic_success"
    assert output["generator_mode"] == "trajectory_dynamic"
    assert output["guardrail"]["mode"] == "trajectory_dynamic"
    assert "src/future_only.py" in output["guardrail"]["sanitized_terms"]
    assert output["items"] == ["Validate relevant implementation detail after the change"]
    assert output["rubric_items"][0]["description"] == "Validate relevant implementation detail after the change"
    assert output["rubric_items"][0]["observable_signal"] == "relevant implementation detail passes focused checks"

from __future__ import annotations

from types import SimpleNamespace

from minisweagent.verifiers.checklist import (
    dedupe_checklist_items,
    generate_issue_checklist,
    infer_checklist_prompt_settings,
    normalize_checklist_items,
    parse_checklist_items,
    parse_checklist_rubric,
    resolve_checklist_output_format,
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

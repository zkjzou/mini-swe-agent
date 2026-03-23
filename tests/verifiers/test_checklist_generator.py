from __future__ import annotations

from types import SimpleNamespace

from minisweagent.verifiers.checklist_generator import (
    generate_trajectory_checklist,
    normalize_checklist_generator_model_config,
    prepare_checklist_generator_template_vars,
    resolve_checklist_generator_prompt_name,
)


def test_prepare_checklist_generator_template_vars_infers_future_steps_from_all_steps():
    template_vars = prepare_checklist_generator_template_vars(
        {
            "task": "Fix bug",
            "steps": [[{"role": "user", "content": "Look at parser.py"}]],
            "all_steps": [
                [{"role": "user", "content": "Look at parser.py"}],
                [{"role": "assistant", "content": "Edit src/new_file.py"}],
            ],
        },
        generator_mode="trajectory_dynamic",
    )

    assert "Look at parser.py" in template_vars["prior_trajectory_text"]
    assert len(template_vars["future_steps"]) == 1
    assert "src/new_file.py" in template_vars["future_steps"][0]


def test_generate_trajectory_checklist_filters_future_only_details_in_dynamic_mode():
    class _DynamicModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: D1\n"
                    "    phase: localize\n"
                    "    weight: 3\n"
                    "    description: Inspect src/new_future_file.py before editing it\n"
                    "    done_when: The agent references src/new_future_file.py\n"
                    "  - id: D2\n"
                    "    phase: diagnose\n"
                    "    weight: 2\n"
                    "    description: Confirm the existing reproduction still matches the reported failure\n"
                    "    done_when: The current failure mode is restated from earlier evidence\n"
                ),
                "extra": {"cost": 0.2},
            }

    config = SimpleNamespace(
        checklist_output_format="rubric_yaml",
        checklist_system_template="unused",
        checklist_prompt_template="unused",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        include_inputs_in_output=True,
        history_message_format="single_prompt",
    )
    output = generate_trajectory_checklist(
        _DynamicModel(),
        config,
        prompt_name="dynamic_success",
        template_vars={
            "task": "Fix parser bug",
            "prior_trajectory_text": "Step 1:\nuser: Reproduce failure in parser.py",
            "future_steps": [
                "Step 2:\nassistant: Edit src/new_future_file.py and run pytest tests/test_new_future_file.py",
            ],
            "generator_mode": "trajectory_dynamic",
        },
    )

    assert output["items"] == [
        "Inspect relevant implementation detail before editing it",
        "Confirm the existing reproduction still matches the reported failure",
    ]
    assert output["generator_prompt_name"] == "dynamic_success"
    assert output["guardrail"]["mode"] == "trajectory_dynamic"
    assert "src/new_future_file.py" in output["guardrail"]["sanitized_terms"]
    assert "Full future steps after the current step" in output["input"]["messages"][-1]["content"]


def test_generate_trajectory_checklist_parses_static_success_checklist_items():
    class _StaticModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "CHECKLIST:\n"
                    "- Localize the faulty parser logic before editing code\n"
                    "- Validate the fix with focused regression checks\n"
                ),
                "extra": {"cost": 0.1},
            }

    config = SimpleNamespace(
        checklist_output_format="list",
        checklist_min_items=2,
        checklist_max_items=8,
        include_inputs_in_output=False,
        history_message_format="single_prompt",
    )
    output = generate_trajectory_checklist(
        _StaticModel(),
        config,
        prompt_name="static_success",
        template_vars={
            "task": "Fix parser bug",
            "full_trajectory_text": "Step 1:\nuser: reproduce\n\nStep 2:\nassistant: localize",
            "generator_mode": "trajectory_success",
        },
    )

    assert output["items"] == [
        "Localize the faulty parser logic before editing code",
        "Validate the fix with focused regression checks",
    ]
    assert output["generator_mode"] == "trajectory_success"
    assert output["checklist_output_format"] == "list"
    assert output["rubric_items"] == []


def test_generate_trajectory_checklist_static_success_supports_multi_turn_chat_context():
    class _StaticModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "CHECKLIST:\n"
                    "- Localize the faulty parser logic before editing code\n"
                    "- Validate the fix with focused regression checks\n"
                ),
                "extra": {"cost": 0.1},
            }

    config = SimpleNamespace(
        checklist_output_format="list",
        checklist_min_items=2,
        checklist_max_items=8,
        include_inputs_in_output=True,
        history_message_format="multi_turn_chat",
    )
    output = generate_trajectory_checklist(
        _StaticModel(),
        config,
        prompt_name="static_success",
        template_vars={
            "task": "Fix parser bug",
            "messages": [
                {"role": "user", "content": "Reproduce the parser bug"},
                {"role": "assistant", "content": "Inspect parser ordering logic"},
            ],
            "all_messages": [
                {"role": "user", "content": "Reproduce the parser bug"},
                {"role": "assistant", "content": "Inspect parser ordering logic"},
            ],
            "generator_mode": "trajectory_success",
        },
    )

    input_messages = output["input"]["messages"]
    assert any(msg["role"] == "assistant" and "Inspect parser ordering logic" in msg["content"] for msg in input_messages)
    assert "Successful trajectory:" in input_messages[-1]["content"]


def test_resolve_checklist_generator_prompt_name_accepts_static_success_v2():
    assert resolve_checklist_generator_prompt_name(
        SimpleNamespace(checklist_generator_prompt_name="static_success_v2")
    ) == "static_success_v2"


def test_resolve_checklist_generator_prompt_name_accepts_static_failure_v2():
    assert resolve_checklist_generator_prompt_name(
        SimpleNamespace(checklist_generator_prompt_name="static_failure_v2")
    ) == "static_failure_v2"


def test_normalize_checklist_generator_model_config_rewrites_litellm_to_textbased():
    normalized = normalize_checklist_generator_model_config({"model_name": "minimax-2.5", "model_class": "litellm"})

    assert normalized["model_class"] == "litellm_textbased"


def test_prepare_checklist_generator_template_vars_sets_dynamic_current_and_successful_trajectory():
    template_vars = prepare_checklist_generator_template_vars(
        {
            "messages": [{"role": "user", "content": "Inspect parser flow"}],
            "all_steps": [
                [{"role": "user", "content": "Inspect parser flow"}],
                [{"role": "assistant", "content": "Patch parser ordering"}],
            ],
        },
        generator_mode="trajectory_dynamic",
    )

    assert template_vars["current_trajectory"] == "user: Inspect parser flow"
    assert "assistant: Patch parser ordering" in template_vars["successful_trajectory_text"]


def test_resolve_checklist_generator_prompt_name_accepts_dynamic_success_v2():
    assert resolve_checklist_generator_prompt_name(
        SimpleNamespace(checklist_generator_prompt_name="dynamic_success_v2")
    ) == "dynamic_success_v2"

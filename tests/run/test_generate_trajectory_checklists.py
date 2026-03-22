import json

from minisweagent.run.utilities.generate_trajectory_checklists import main


def test_generate_trajectory_checklists_dynamic_mode_from_trajectory_file(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "sample.traj.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "problem_statement": "Fix validation flow",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Fix validation flow"},
                    {"role": "assistant", "content": "Inspect current validation logic"},
                    {"role": "user", "content": "Observed issue in src/current.py"},
                    {"role": "assistant", "content": "Edit src/future_only.py"},
                    {"role": "user", "content": "Focused test now passes"},
                ],
            }
        )
    )
    output_path = tmp_path / "generated.jsonl"

    class _DynamicRubricModel:
        def query(self, messages, **kwargs):
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
            }

    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.get_model",
        lambda **kwargs: _DynamicRubricModel(),
    )

    main(
        input_path=None,
        trajectory=trajectory_path,
        output_path=output_path,
        generator_mode="trajectory_dynamic",
        prompt_name="",
        compare_trajectory=None,
        existing_rubric=None,
        step_index=1,
        model_name="deterministic",
        model_class="deterministic",
        config_spec=[],
    )

    row = json.loads(output_path.read_text().strip())
    prompt = row["output"]["input"]["messages"][-1]["content"]

    assert "Full future steps after the current step" in prompt
    assert "Edit src/future_only.py" in prompt
    assert row["output"]["generator_prompt_name"] == "dynamic_success"
    assert row["output"]["guardrail"]["mode"] == "trajectory_dynamic"
    assert row["output"]["items"] == ["Validate relevant implementation detail after the change"]


def test_generate_trajectory_checklists_can_enable_langfuse(tmp_path, monkeypatch):
    input_path = tmp_path / "rows.json"
    input_path.write_text(json.dumps([{"task": "Fix bug", "messages": [{"role": "user", "content": "Fix bug"}]}]))
    output_path = tmp_path / "generated.jsonl"
    calls = {}

    class _StaticModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: S1\n"
                    "    phase: understand\n"
                    "    weight: 3\n"
                    "    description: Understand the issue before editing\n"
                    "    done_when: The issue goal is restated accurately\n"
                ),
            }

    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.enable_langfuse_tracing",
        lambda: calls.update({"enabled": True}),
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.make_langfuse_session_id",
        lambda **kwargs: "session-123",
    )

    def _fake_attach(config, *, session_id):
        calls.update({"session_id": session_id, "config": config})
        model_config = config.setdefault("model", {})
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["litellm_session_id"] = session_id

    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.attach_langfuse_session_metadata",
        _fake_attach,
    )
    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.get_model",
        lambda **kwargs: _StaticModel(),
    )

    main(
        input_path=input_path,
        trajectory=None,
        output_path=output_path,
        generator_mode="trajectory_success",
        prompt_name="static_success_v2",
        compare_trajectory=None,
        existing_rubric=None,
        step_index=None,
        model_name="minimax-2.5",
        model_class="litellm",
        enable_langfuse=True,
        config_spec=[],
    )

    assert calls["enabled"] is True
    assert calls["session_id"] == "session-123"
    assert calls["config"]["model"]["model_kwargs"]["litellm_session_id"] == "session-123"


def test_generate_trajectory_checklists_strips_coding_wrapper_from_issue_description(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "wrapped.traj.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": (
                            "Consider the following PR description:\n"
                            "<pr_description>\nFix validation flow for empty payloads\n</pr_description>\n\n"
                            "You'll be helping implement necessary changes to meet requirements in the PR description.\n"
                            "Your task is specifically to make changes to non-test files in the current directory."
                        ),
                    },
                ],
            }
        )
    )
    output_path = tmp_path / "generated.jsonl"

    class _StaticModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": (
                    "rubric:\n"
                    "  - id: S1\n"
                    "    phase: understand\n"
                    "    weight: 3\n"
                    "    description: Understand the issue before editing\n"
                    "    done_when: The issue goal is restated accurately\n"
                ),
            }

    monkeypatch.setattr(
        "minisweagent.run.utilities.generate_trajectory_checklists.get_model",
        lambda **kwargs: _StaticModel(),
    )

    main(
        input_path=None,
        trajectory=trajectory_path,
        output_path=output_path,
        generator_mode="trajectory_success",
        prompt_name="static_success_v2",
        compare_trajectory=None,
        existing_rubric=None,
        step_index=None,
        model_name="deterministic",
        model_class="deterministic",
        config_spec=[],
    )

    row = json.loads(output_path.read_text().strip())
    prompt = row["output"]["input"]["messages"][-1]["content"]

    assert "Fix validation flow for empty payloads" in prompt
    assert "Consider the following PR description" not in prompt
    assert "You'll be helping implement necessary changes" not in prompt
    assert "Your task is specifically to make changes" not in prompt

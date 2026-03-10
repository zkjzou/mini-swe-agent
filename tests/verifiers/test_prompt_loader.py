from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import minisweagent.verifiers.prompt_loader as prompt_loader_module
import pytest
from minisweagent.verifiers.prompt_loader import (
    _parse_reward_prompt_file,
    _parse_selection_prompt_file,
    apply_prompt_overrides,
)


def test_parse_selection_prompt_file_with_markers():
    content = """[[[SYSTEM_TEMPLATE]]]
system content

[[[SELECTION_TEMPLATE]]]
selection content
"""
    system_template, selection_prompt = _parse_selection_prompt_file(content)
    assert system_template == "system content"
    assert selection_prompt == "selection content"


def test_parse_selection_prompt_file_legacy():
    content = "selection only content"
    system_template, selection_prompt = _parse_selection_prompt_file(content)
    assert system_template is None
    assert selection_prompt == "selection only content"


def test_parse_reward_prompt_file_with_markers():
    content = """[[[SYSTEM_TEMPLATE]]]
system content

[[[REWARD_PROMPT_TEMPLATE]]]
reward content
"""
    system_template, reward_prompt = _parse_reward_prompt_file(content)
    assert system_template == "system content"
    assert reward_prompt == "reward content"


def test_parse_reward_prompt_file_legacy():
    content = "reward only content"
    system_template, reward_prompt = _parse_reward_prompt_file(content)
    assert system_template is None
    assert reward_prompt == "reward only content"


def test_apply_prompt_overrides_reward_single_file_with_system(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "reward.jinja").write_text(
        "[[[SYSTEM_TEMPLATE]]]\nloaded system\n[[[REWARD_PROMPT_TEMPLATE]]]\nloaded reward prompt\n"
    )
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="reward_model",
        system_template="original system",
        reward_system_template="original reward system",
        reward_prompt_template="original reward prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.reward_system_template == "loaded system"
    assert updated.reward_prompt_template == "loaded reward prompt"
    assert updated.checklist_system_template == "original checklist system"
    assert updated.checklist_prompt_template == "original checklist prompt"


def test_apply_prompt_overrides_llm_single_file_with_system(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "selection.jinja").write_text(
        "[[[SYSTEM_TEMPLATE]]]\nloaded llm system\n[[[SELECTION_TEMPLATE]]]\nloaded selection prompt\n"
    )
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="llm",
        system_template="original system",
        selection_template="original selection prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.system_template == "loaded llm system"
    assert updated.selection_template == "loaded selection prompt"
    assert updated.checklist_system_template == "original checklist system"
    assert updated.checklist_prompt_template == "original checklist prompt"


def test_apply_prompt_overrides_llm_single_file_legacy_prompt_only(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "selection.jinja").write_text("legacy selection prompt")
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="llm",
        system_template="original system",
        selection_template="original selection prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.system_template == "original system"
    assert updated.selection_template == "legacy selection prompt"
    assert updated.checklist_system_template == "original checklist system"
    assert updated.checklist_prompt_template == "original checklist prompt"


@pytest.mark.parametrize(
    "path",
    [
        "prompts/verifier/basic/verifier/selection.jinja",
        "prompts/verifier/basic/reward/reward.jinja",
        "prompts/verifier/basic_mini/verifier/selection.jinja",
        "prompts/verifier/basic_mini/reward/reward.jinja",
        "prompts/verifier/checklist/verifier/selection.jinja",
        "prompts/verifier/checklist/reward/reward.jinja",
        "prompts/verifier/checklist_v2/verifier/selection.jinja",
        "prompts/verifier/checklist_v2/reward/reward.jinja",
        "prompts/verifier/dynamic_checklist_modify/verifier/selection.jinja",
        "prompts/verifier/dynamic_checklist_modify/reward/reward.jinja",
        "prompts/verifier/dynamic_checklist_regenerate/verifier/selection.jinja",
        "prompts/verifier/dynamic_checklist_regenerate/reward/reward.jinja",
    ],
)
def test_core_verifier_prompts_keep_task_out_of_system_section(path: str) -> None:
    content = Path(path).read_text()
    system_start = content.index("[[[SYSTEM_TEMPLATE]]]")
    next_markers = [
        content.find(marker)
        for marker in (
            "[[[SELECTION_TEMPLATE]]]",
            "[[[REWARD_PROMPT_TEMPLATE]]]",
            "[[[CHECKLIST_SYSTEM_TEMPLATE]]]",
            "[[[CHECKLIST_PROMPT_TEMPLATE]]]",
        )
        if content.find(marker) != -1
    ]
    system_end = min(next_markers) if next_markers else len(content)

    assert "Task: {{ task }}" not in content[system_start:system_end]
    assert "Task: {{ task }}" in content[system_end:]


def test_apply_prompt_overrides_reward_single_file_legacy_prompt_only(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "reward.jinja").write_text("legacy reward prompt")
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="reward_model",
        system_template="original system",
        reward_system_template="original reward system",
        reward_prompt_template="original reward prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.reward_system_template == "original reward system"
    assert updated.reward_prompt_template == "legacy reward prompt"
    assert updated.checklist_system_template == "original checklist system"
    assert updated.checklist_prompt_template == "original checklist prompt"


def test_apply_prompt_overrides_llm_loads_checklist_markers(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "selection.jinja").write_text(
        "[[[SYSTEM_TEMPLATE]]]\n"
        "loaded llm system\n"
        "[[[SELECTION_TEMPLATE]]]\n"
        "loaded selection prompt\n"
        "[[[CHECKLIST_SYSTEM_TEMPLATE]]]\n"
        "loaded checklist system\n"
        "[[[CHECKLIST_PROMPT_TEMPLATE]]]\n"
        "loaded checklist prompt\n"
    )
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="llm",
        system_template="original system",
        selection_template="original selection prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.system_template == "loaded llm system"
    assert updated.selection_template == "loaded selection prompt"
    assert updated.checklist_system_template == "loaded checklist system"
    assert updated.checklist_prompt_template == "loaded checklist prompt"


def test_apply_prompt_overrides_reward_loads_checklist_markers(tmp_path):
    prompt_dir = tmp_path / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "reward.jinja").write_text(
        "[[[SYSTEM_TEMPLATE]]]\n"
        "loaded reward system\n"
        "[[[REWARD_PROMPT_TEMPLATE]]]\n"
        "loaded reward prompt\n"
        "[[[CHECKLIST_SYSTEM_TEMPLATE]]]\n"
        "loaded checklist system\n"
        "[[[CHECKLIST_PROMPT_TEMPLATE]]]\n"
        "loaded checklist prompt\n"
    )
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir=str(tmp_path / "prompts" / "verifier"),
        verifier_type="reward_model",
        reward_system_template="original reward system",
        reward_prompt_template="original reward prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.reward_system_template == "loaded reward system"
    assert updated.reward_prompt_template == "loaded reward prompt"
    assert updated.checklist_system_template == "loaded checklist system"
    assert updated.checklist_prompt_template == "loaded checklist prompt"


def test_apply_prompt_overrides_resolves_relative_prompt_dir_from_repo_root(tmp_path, monkeypatch):
    fake_repo_root = tmp_path / "repo"
    prompt_dir = fake_repo_root / "prompts" / "verifier" / "custom"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "selection.jinja").write_text("resolved from repo root")
    fake_cwd = tmp_path / "outside_repo"
    fake_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(fake_cwd)
    monkeypatch.setattr(prompt_loader_module, "package_dir", fake_repo_root / "src" / "minisweagent")
    config = SimpleNamespace(
        prompt_name="custom",
        prompt_dir="prompts/verifier",
        verifier_type="llm",
        system_template="original system",
        selection_template="original selection prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert updated.selection_template == "resolved from repo root"


def test_apply_prompt_overrides_loads_builtin_basic_mini_verifier_prompt():
    config = SimpleNamespace(
        prompt_name="basic_mini/verifier",
        prompt_dir="prompts/verifier",
        verifier_type="llm",
        system_template="original system",
        selection_template="original selection prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert "choose the single best candidate action" in updated.system_template.lower()
    assert "choose the single best candidate action" in updated.selection_template.lower()
    assert "Task: {{ task }}" not in updated.system_template
    assert "Task: {{ task }}" in updated.selection_template
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" in updated.system_template
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" in updated.selection_template
    assert "Candidates:" in updated.selection_template


def test_apply_prompt_overrides_loads_builtin_basic_mini_reward_prompt():
    config = SimpleNamespace(
        prompt_name="basic_mini/reward",
        prompt_dir="prompts/verifier",
        verifier_type="reward_model",
        reward_system_template="original reward system",
        reward_prompt_template="original reward prompt",
        checklist_system_template="original checklist system",
        checklist_prompt_template="original checklist prompt",
    )

    updated = apply_prompt_overrides(config)

    assert "evaluate a single candidate next action" in updated.reward_system_template.lower()
    assert "evaluate a single candidate next action" in updated.reward_prompt_template.lower()
    assert "Task: {{ task }}" not in updated.reward_system_template
    assert "Task: {{ task }}" in updated.reward_prompt_template
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" in updated.reward_system_template
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt" in updated.reward_prompt_template
    assert "Candidate action:" in updated.reward_prompt_template

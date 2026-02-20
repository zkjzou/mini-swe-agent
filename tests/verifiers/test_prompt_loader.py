from __future__ import annotations

from types import SimpleNamespace

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

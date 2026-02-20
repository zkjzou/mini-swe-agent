from __future__ import annotations

from pathlib import Path
from typing import Any

_SYSTEM_MARKER = "[[[SYSTEM_TEMPLATE]]]"
_SELECTION_MARKER = "[[[SELECTION_TEMPLATE]]]"
_REWARD_MARKER = "[[[REWARD_PROMPT_TEMPLATE]]]"
_CHECKLIST_SYSTEM_MARKER = "[[[CHECKLIST_SYSTEM_TEMPLATE]]]"
_CHECKLIST_PROMPT_MARKER = "[[[CHECKLIST_PROMPT_TEMPLATE]]]"


def apply_prompt_overrides(config: Any) -> Any:
    """Load verifier prompts from prompts/verifier/<prompt_name> if configured."""
    prompt_name = getattr(config, "prompt_name", None)
    if not prompt_name:
        return config
    prompt_dir = Path(getattr(config, "prompt_dir", "prompts/verifier"))
    if not prompt_dir.is_absolute():
        prompt_dir = Path.cwd() / prompt_dir
    prompt_root = prompt_dir / prompt_name
    system_path = prompt_root / "system.jinja"

    if config.verifier_type == "llm":
        selection_path = prompt_root / "selection.jinja"
        if selection_path.is_file():
            selection_content = selection_path.read_text()
            sections = _parse_prompt_sections(selection_content)
            if _SELECTION_MARKER in sections:
                if _SYSTEM_MARKER in sections:
                    config.system_template = sections[_SYSTEM_MARKER]
                elif system_path.is_file():
                    config.system_template = system_path.read_text()
                config.selection_template = sections[_SELECTION_MARKER]
                _apply_checklist_overrides(config, sections)
            else:
                if system_path.is_file():
                    config.system_template = system_path.read_text()
                config.selection_template = selection_content
        elif system_path.is_file():
            config.system_template = system_path.read_text()
    elif config.verifier_type in {"reward_model", "world_model"}:
        reward_path = prompt_root / "reward.jinja"
        if reward_path.is_file():
            reward_content = reward_path.read_text()
            sections = _parse_prompt_sections(reward_content)
            if _REWARD_MARKER in sections:
                if _SYSTEM_MARKER in sections:
                    config.reward_system_template = sections[_SYSTEM_MARKER]
                elif system_path.is_file():
                    config.reward_system_template = system_path.read_text()
                config.reward_prompt_template = sections[_REWARD_MARKER]
                _apply_checklist_overrides(config, sections)
            else:
                if system_path.is_file():
                    config.reward_system_template = system_path.read_text()
                config.reward_prompt_template = reward_content
        elif system_path.is_file():
            config.reward_system_template = system_path.read_text()

    return config


def _parse_selection_prompt_file(content: str) -> tuple[str | None, str]:
    """Parse selection prompt file."""
    return _parse_single_prompt_file(content, section_marker=_SELECTION_MARKER)


def _parse_reward_prompt_file(content: str) -> tuple[str | None, str]:
    """Parse reward prompt file."""
    return _parse_single_prompt_file(content, section_marker=_REWARD_MARKER)


def _parse_single_prompt_file(content: str, *, section_marker: str) -> tuple[str | None, str]:
    """Parse a single-file prompt.

    Supported formats:
    1) Single-file format with markers:
       [[[SYSTEM_TEMPLATE]]]
       ...
       [[[<SECTION>]]]
       ...
    2) Legacy format: file contains only the section template.
    """
    sections = _parse_prompt_sections(content)
    if _SYSTEM_MARKER not in sections or section_marker not in sections:
        return None, content

    marker_order = sorted(
        [
            (content.find(_SYSTEM_MARKER), _SYSTEM_MARKER),
            (content.find(section_marker), section_marker),
        ],
        key=lambda x: x[0],
    )
    if marker_order[0][1] != _SYSTEM_MARKER:
        return None, content
    return sections[_SYSTEM_MARKER], sections[section_marker]


def _apply_checklist_overrides(config: Any, sections: dict[str, str]) -> None:
    if _CHECKLIST_SYSTEM_MARKER in sections:
        config.checklist_system_template = sections[_CHECKLIST_SYSTEM_MARKER]
    if _CHECKLIST_PROMPT_MARKER in sections:
        config.checklist_prompt_template = sections[_CHECKLIST_PROMPT_MARKER]


def _parse_prompt_sections(content: str) -> dict[str, str]:
    markers = [
        _SYSTEM_MARKER,
        _SELECTION_MARKER,
        _REWARD_MARKER,
        _CHECKLIST_SYSTEM_MARKER,
        _CHECKLIST_PROMPT_MARKER,
    ]
    positions = [(content.find(marker), marker) for marker in markers if content.find(marker) != -1]
    if not positions:
        return {}
    positions.sort(key=lambda item: item[0])

    sections: dict[str, str] = {}
    for idx, (start, marker) in enumerate(positions):
        section_start = start + len(marker)
        section_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(content)
        sections[marker] = content[section_start:section_end].strip()
    return sections

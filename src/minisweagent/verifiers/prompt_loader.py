from __future__ import annotations

from pathlib import Path
from typing import Any


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

    if system_path.is_file():
        config.system_template = system_path.read_text()

    if config.verifier_type == "llm":
        selection_path = prompt_root / "selection.jinja"
        if selection_path.is_file():
            config.selection_template = selection_path.read_text()
    elif config.verifier_type == "reward_model":
        reward_path = prompt_root / "reward.jinja"
        if reward_path.is_file():
            config.reward_prompt_template = reward_path.read_text()

    return config

from __future__ import annotations

from typing import Any

from minisweagent.verifiers.checklist import infer_checklist_prompt_settings
from minisweagent.verifiers.prompt_loader import apply_prompt_overrides


def resolve_verifier_runtime_config(config: Any) -> Any:
    """Apply prompt-name inference and prompt-template overrides to a verifier config."""
    if (
        getattr(config, "checklist_mode", None) == "issue_progress"
        and bool(getattr(config, "checklist_dynamic", False))
        and getattr(config, "verifier_type", None) in {"llm", "reward_model"}
        and not getattr(config, "prompt_name", None)
    ):
        suffix = "verifier" if getattr(config, "verifier_type", None) == "llm" else "reward"
        config.prompt_name = f"dynamic_checklist_{getattr(config, 'checklist_update_mode', 'regenerate')}/{suffix}"

    inferred_checklist_settings = infer_checklist_prompt_settings(getattr(config, "prompt_name", None))
    if inferred_checklist_settings is not None:
        config.checklist_mode = inferred_checklist_settings["checklist_mode"]
        config.checklist_dynamic = inferred_checklist_settings["checklist_dynamic"]
        config.checklist_update_mode = inferred_checklist_settings["checklist_update_mode"]

    return apply_prompt_overrides(config)


def verifier_uses_checklist_mode(config: Any) -> bool:
    return (
        getattr(config, "checklist_mode", None) == "issue_progress"
        and getattr(config, "verifier_type", None) in {"llm", "reward_model"}
    )

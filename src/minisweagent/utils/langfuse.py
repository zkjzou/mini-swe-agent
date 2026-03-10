from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import litellm


def enable_langfuse_tracing() -> None:
    callbacks = getattr(litellm, "callbacks", None)
    if isinstance(callbacks, list):
        if "langfuse_otel" not in callbacks:
            callbacks.append("langfuse_otel")
        return
    litellm.callbacks = ["langfuse_otel"]


def make_langfuse_session_id(*, prefix: str, output_path: Path, parts: Iterable[str] = ()) -> str:
    session_parts = [prefix]
    for part in parts:
        cleaned = str(part or "").strip()
        if cleaned:
            session_parts.append(cleaned)
    session_parts.append(output_path.name or "run")
    session_parts.append(str(int(time.time())))
    return ":".join(session_parts)


def attach_langfuse_session_metadata(config: dict, *, session_id: str) -> None:
    def _apply_to_model_config(model_config: dict | None) -> None:
        if not isinstance(model_config, dict):
            return
        model_kwargs = model_config.setdefault("model_kwargs", {})
        if not isinstance(model_kwargs, dict):
            return
        metadata = model_kwargs.get("metadata")
        if metadata is None:
            metadata = {}
            model_kwargs["metadata"] = metadata
        if not isinstance(metadata, dict):
            return
        metadata["session_id"] = session_id
        model_kwargs["litellm_session_id"] = session_id

    _apply_to_model_config(config.get("model"))
    agent_config = config.get("agent")
    if isinstance(agent_config, dict):
        verifier_config = agent_config.get("verifier")
        if isinstance(verifier_config, dict):
            _apply_to_model_config(verifier_config.get("model"))

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


def analyze_action_similarity(
    candidates: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.9,
    metric: str = "token_jaccard",
) -> dict[str, Any]:
    """Analyze pairwise similarity among candidate actions."""
    signatures = [_candidate_signature(candidate) for candidate in candidates]
    pairwise: list[dict[str, Any]] = []
    min_similarity = 1.0

    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            similarity = _similarity(signatures[i], signatures[j], metric=metric)
            pairwise.append({"left": i, "right": j, "similarity": similarity})
            min_similarity = min(min_similarity, similarity)

    if len(candidates) < 2:
        return {
            "metric": metric,
            "threshold": float(threshold),
            "min_pairwise_similarity": 1.0,
            "pairwise_similarity": [],
            "should_skip_verifier": False,
        }

    should_skip = all(item["similarity"] >= threshold for item in pairwise)
    return {
        "metric": metric,
        "threshold": float(threshold),
        "min_pairwise_similarity": min_similarity,
        "pairwise_similarity": pairwise,
        "should_skip_verifier": should_skip,
    }


def _candidate_signature(candidate: dict[str, Any]) -> set[str]:
    actions = candidate.get("actions") or []
    if not isinstance(actions, list):
        return set()
    commands: list[str] = []
    for action in actions:
        if isinstance(action, dict):
            command = action.get("command")
            if isinstance(command, str) and command:
                commands.append(command)
    text = "\n".join(commands)
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return set(normalized.split()) if normalized else set()


def _similarity(left: set[str], right: set[str], *, metric: str) -> float:
    if metric != "token_jaccard":
        raise ValueError(f"Unknown action similarity metric: {metric}")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

from __future__ import annotations

from minisweagent.verifiers.action_similarity import analyze_action_similarity


def test_analyze_action_similarity_identical_actions_skips_verifier():
    candidates = [
        {"actions": [{"command": "python -m pytest -q"}]},
        {"actions": [{"command": "python -m pytest -q"}]},
        {"actions": [{"command": "python -m pytest -q"}]},
    ]
    result = analyze_action_similarity(candidates, threshold=0.9, metric="token_jaccard")
    assert result["should_skip_verifier"] is True
    assert result["min_pairwise_similarity"] == 1.0
    assert len(result["pairwise_similarity"]) == 3


def test_analyze_action_similarity_different_actions_does_not_skip():
    candidates = [
        {"actions": [{"command": "python -m pytest -q"}]},
        {"actions": [{"command": "git status"}]},
    ]
    result = analyze_action_similarity(candidates, threshold=0.9, metric="token_jaccard")
    assert result["should_skip_verifier"] is False
    assert result["min_pairwise_similarity"] < 0.9

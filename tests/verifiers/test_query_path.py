from __future__ import annotations

from types import SimpleNamespace

from minisweagent.verifiers.llm import LLMVerifier
from minisweagent.verifiers.reward_model import RewardModelVerifier


class _RawQueryModel:
    def __init__(self):
        self.query_called = False
        self.prepared_messages = None
        self.raw_messages = None

    def query(self, messages, **kwargs):
        self.query_called = True
        raise AssertionError("query() should not be called for verifier text queries when _query is available")

    def _prepare_messages_for_api(self, messages):
        self.prepared_messages = messages
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def _query(self, messages, **kwargs):
        self.raw_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "REASONING: candidate 2 is best.\n"
                            "SCORES:\n"
                            "- Candidate 1: 0.2\n"
                            "- Candidate 2: 0.9\n"
                            "CHECKLIST_ITEM_SCORES:\n"
                            "- Item 1: 0.4\n"
                            "- Item 2: 0.8\n"
                            "PROGRESS: 0.6\n"
                            "FINAL: 2"
                        )
                    }
                }
            ]
        }

    def _calculate_cost(self, response):
        return {"cost": 0.42}


class _RawResponsesModel:
    def __init__(self):
        self.query_called = False

    def query(self, messages, **kwargs):
        self.query_called = True
        raise AssertionError("query() should not be called for verifier text queries when _query is available")

    def _prepare_messages_for_api(self, messages):
        return messages

    def _query(self, messages, **kwargs):
        reward = "0.9" if "Option 2" in messages[-1]["content"] else "0.2"
        progress = "0.7" if "Option 2" in messages[-1]["content"] else "0.3"
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                f"CHECKLIST_ITEM_SCORES:\n- Item 1: {progress}\n- Item 2: 0.5\n"
                                f"PROGRESS: {progress}\nREWARD: {reward}"
                            ),
                        }
                    ],
                }
            ]
        }

    def _calculate_cost(self, response):
        return {"cost": 0.3}


class _QueryOnlyModel:
    def query(self, messages, **kwargs):
        return {
            "role": "assistant",
            "content": "SCORES:\n- Candidate 1: 0.3\n- Candidate 2: 0.8\nFINAL: 2",
            "extra": {"cost": 0.7},
        }


def test_llm_verifier_uses_raw_query_path_when_available():
    model = _RawQueryModel()
    config = SimpleNamespace(
        system_template="system",
        selection_template="Candidates:\n{% for c in candidates %}{{ c.index + selection_index_base }}. {{ c.content }}\n{% endfor %}",
        selection_index_base=1,
        selection_regex=r"(\d+)",
        fallback="first_candidate",
    )
    verifier = LLMVerifier(model, config)
    candidates = [
        {"index": 0, "content": "Option 1", "action": "echo first"},
        {"index": 1, "content": "Option 2", "action": "echo second"},
    ]

    selected_index, metadata = verifier.select(
        candidates=candidates,
        template_vars={"checklist_items": ["reproduce", "validate"]},
    )

    assert selected_index == 1
    assert metadata["response_cost"] == 0.42
    assert "FINAL: 2" in metadata["raw_output"]
    assert metadata["scores"] == [0.2, 0.9]
    assert metadata["checklist_item_scores"] == [0.4, 0.8]
    assert metadata["progress_score"] == 0.6
    assert model.query_called is False
    assert model.prepared_messages is not None
    assert model.raw_messages is not None


def test_reward_verifier_uses_raw_query_path_and_extracts_response_output_text():
    model = _RawResponsesModel()
    config = SimpleNamespace(
        reward_system_template="system",
        reward_prompt_template="Candidate action:\n{{ candidate.content }}",
        reward_regex=r"REWARD:\s*([+-]?\d+(?:\.\d+)?)",
        fallback="first_candidate",
    )
    verifier = RewardModelVerifier(model, config)
    candidates = [
        {"index": 0, "content": "Option 1", "action": "echo first"},
        {"index": 1, "content": "Option 2", "action": "echo second"},
    ]

    selected_index, metadata = verifier.select(
        candidates=candidates,
        template_vars={"checklist_items": ["reproduce", "validate"]},
    )

    assert selected_index == 1
    assert metadata["rewards"] == [0.2, 0.9]
    assert metadata["candidate_progress_scores"] == [0.3, 0.7]
    assert metadata["candidate_checklist_item_scores"] == [[0.3, 0.5], [0.7, 0.5]]
    assert metadata["response_costs"] == [0.3, 0.3]
    assert "REWARD: 0.2" in metadata["raw_outputs"][0]
    assert "REWARD: 0.9" in metadata["raw_outputs"][1]
    assert model.query_called is False


def test_llm_verifier_falls_back_to_query_when_raw_query_path_not_available():
    model = _QueryOnlyModel()
    config = SimpleNamespace(
        system_template="system",
        selection_template="Pick one candidate.",
        selection_index_base=1,
        selection_regex=r"(\d+)",
        fallback="first_candidate",
    )
    verifier = LLMVerifier(model, config)
    candidates = [
        {"index": 0, "content": "Option 1", "action": "echo first"},
        {"index": 1, "content": "Option 2", "action": "echo second"},
    ]

    selected_index, metadata = verifier.select(candidates=candidates)

    assert selected_index == 1
    assert metadata["response_cost"] == 0.7
    assert "FINAL: 2" in metadata["raw_output"]
    assert metadata["scores"] == [0.3, 0.8]

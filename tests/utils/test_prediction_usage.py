from __future__ import annotations

from pathlib import Path

from minisweagent.utils import prediction_usage


def _analyze_payload(payload: dict, tmp_path: Path) -> dict:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    path = run_dir / "instance.traj.json"
    return prediction_usage._analyze_payload(payload, path, root=tmp_path)


def test_prediction_usage_assigns_all_tokens_to_agent_when_no_verifier(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        }
                    }
                },
            }
        ]
    }

    metrics = _analyze_payload(payload, tmp_path)
    assert metrics["prompt_tokens"] == 10
    assert metrics["completion_tokens"] == 4
    assert metrics["total_tokens"] == 14
    assert metrics["agent_prompt_tokens"] == 10
    assert metrics["agent_completion_tokens"] == 4
    assert metrics["agent_total_tokens"] == 14
    assert metrics["verifier_prompt_tokens"] == 0
    assert metrics["verifier_completion_tokens"] == 0
    assert metrics["verifier_total_tokens"] == 0


def test_prediction_usage_splits_llm_verifier_and_checklist_tokens(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 40,
                            "total_tokens": 140,
                        }
                    },
                    "verifier": {
                        "verifier_output": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 30,
                                    "completion_tokens": 10,
                                    "total_tokens": 40,
                                }
                            },
                            "checklist": {
                                "response": {
                                    "usage": {
                                        "prompt_tokens": 8,
                                        "completion_tokens": 2,
                                        "total_tokens": 10,
                                    }
                                }
                            },
                        }
                    },
                },
            }
        ]
    }

    metrics = _analyze_payload(payload, tmp_path)
    assert metrics["agent_prompt_tokens"] == 100
    assert metrics["agent_completion_tokens"] == 40
    assert metrics["agent_total_tokens"] == 140
    assert metrics["verifier_prompt_tokens"] == 38
    assert metrics["verifier_completion_tokens"] == 12
    assert metrics["verifier_total_tokens"] == 50
    assert metrics["prompt_tokens"] == 138
    assert metrics["completion_tokens"] == 52
    assert metrics["total_tokens"] == 190


def test_prediction_usage_sums_reward_verifier_responses_tokens(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 10,
                            "total_tokens": 60,
                        }
                    },
                    "verifier": {
                        "verifier_output": {
                            "responses": [
                                {
                                    "usage": {
                                        "prompt_tokens": 20,
                                        "completion_tokens": 5,
                                        "total_tokens": 25,
                                    }
                                },
                                {
                                    "extra": {
                                        "response": {
                                            "usage": {
                                                "prompt_tokens": 24,
                                                "completion_tokens": 6,
                                                "total_tokens": 30,
                                            }
                                        }
                                    }
                                },
                            ]
                        }
                    },
                },
            }
        ]
    }

    metrics = _analyze_payload(payload, tmp_path)
    assert metrics["agent_prompt_tokens"] == 50
    assert metrics["agent_completion_tokens"] == 10
    assert metrics["agent_total_tokens"] == 60
    assert metrics["verifier_prompt_tokens"] == 44
    assert metrics["verifier_completion_tokens"] == 11
    assert metrics["verifier_total_tokens"] == 55
    assert metrics["prompt_tokens"] == 94
    assert metrics["completion_tokens"] == 21
    assert metrics["total_tokens"] == 115


def test_prediction_usage_handles_missing_total_tokens_fields(tmp_path):
    payload = {
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {"usage": {"prompt_tokens": 12, "completion_tokens": 8}},
                    "verifier": {
                        "verifier_output": {
                            "response": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                        }
                    },
                },
            }
        ]
    }

    metrics = _analyze_payload(payload, tmp_path)
    assert metrics["agent_total_tokens"] == 20
    assert metrics["verifier_total_tokens"] == 10
    assert metrics["total_tokens"] == 30


def test_prediction_usage_reconciles_split_with_model_stats_totals(tmp_path):
    payload = {
        "info": {
            "model_stats": {
                "tokens_sent": 200,
                "tokens_received": 100,
                "total_tokens": 300,
            }
        },
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 20,
                            "total_tokens": 70,
                        }
                    },
                    "verifier": {
                        "verifier_output": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 4,
                                    "total_tokens": 14,
                                }
                            }
                        }
                    },
                },
            }
        ],
    }

    metrics = _analyze_payload(payload, tmp_path)
    assert metrics["prompt_tokens"] == 200
    assert metrics["completion_tokens"] == 100
    assert metrics["total_tokens"] == 300
    assert metrics["agent_prompt_tokens"] + metrics["verifier_prompt_tokens"] == metrics["prompt_tokens"]
    assert metrics["agent_completion_tokens"] + metrics["verifier_completion_tokens"] == metrics["completion_tokens"]
    assert metrics["agent_total_tokens"] + metrics["verifier_total_tokens"] == metrics["total_tokens"]

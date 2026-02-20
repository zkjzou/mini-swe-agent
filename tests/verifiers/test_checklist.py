from __future__ import annotations

from types import SimpleNamespace

from minisweagent.verifiers.checklist import (
    generate_issue_checklist,
    normalize_checklist_items,
    parse_checklist_items,
)


def test_parse_checklist_items_from_numbered_lines():
    content = "CHECKLIST:\n1. Reproduce bug\n2) Implement fix\n- Run targeted tests\n"
    items = parse_checklist_items(content, item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")
    assert items == ["Reproduce bug", "Implement fix", "Run targeted tests"]


def test_normalize_checklist_items_dedupes_and_fills_min_items():
    items = [
        "Reproduce bug",
        "Reproduce   bug",
        "Implement fix",
    ]
    normalized = normalize_checklist_items(items, min_items=4, max_items=5)
    assert normalized[0] == "Reproduce bug"
    assert normalized[1] == "Implement fix"
    assert len(normalized) == 4


def test_generate_issue_checklist_uses_model_query_and_parses_items():
    class _QueryOnlyModel:
        def query(self, messages, **kwargs):
            return {
                "role": "assistant",
                "content": "CHECKLIST:\n- Reproduce issue\n- Patch source code\n- Run tests\n",
                "extra": {"cost": 0.25},
            }

    config = SimpleNamespace(
        checklist_system_template="system",
        checklist_prompt_template="task: {{ task }}",
        checklist_item_regex=r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$",
        checklist_min_items=3,
        checklist_max_items=5,
    )
    output = generate_issue_checklist(_QueryOnlyModel(), config, template_vars={"task": "sample issue", "messages": []})

    assert output["items"] == ["Reproduce issue", "Patch source code", "Run tests"]
    assert output["response_cost"] == 0.25
    assert "CHECKLIST" in output["raw_output"]

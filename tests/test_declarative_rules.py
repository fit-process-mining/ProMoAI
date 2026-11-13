import pytest

from powl.discovery.rule_based import (
    AtMostOnceRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)

from promoai.general_utils.llm_connection import LLMConnection
from promoai.prompting import declarative_rules


@pytest.mark.parametrize(
    "rule_code,expected",
    [
        ("AtMostOnce('A')", AtMostOnceRule("A")),
        ("Init('A')", InitializationRule("A")),
        ("InitializationRule('A')", InitializationRule("A")),
        ("End('B')", EndRule("B")),
        ("Existence('A')", ExistenceRule("A")),
        ("CoExistence('A', 'B')", CoExistenceRule("A", "B")),
        ("NotCoExistence('A', 'B')", NotCoExistenceRule("A", "B")),
        ("NotSuccession('A', 'B')", NotSuccessionRule("A", "B")),
        ("Precedence('A', 'B')", PrecedenceRule("A", "B")),
        ("ChainPrecedence('A', 'B')", ChainPrecedenceRule("A", "B")),
        ("RespondedExistence('A', 'B')", RespondedExistenceRule("A", "B")),
        ("Response('A', 'B')", ResponseRule("A", "B")),
        ("ChainResponse('A', 'B')", ChainResponseRule("A", "B")),
    ],
)
def test_parse_each_rule(rule_code, expected):
    code, rules = declarative_rules.parse_declare_rules(rule_code, ["A", "B"])
    assert rules == [expected]
    assert code.startswith("rule1 = ")


def test_parse_fenced_rule_assignments_and_deduplicate():
    source = """```python
rule_1 = Init("A")
rule_2 = Response("A", "B")
rule_3 = Response("A", "B")
```"""
    code, rules = declarative_rules.parse_declare_rules(source, ["A", "B"])
    assert rules == [InitializationRule("A"), ResponseRule("A", "B")]
    assert "rule3" not in code
    assert "Init('A')" in code


def test_activity_labels_with_quotes_are_preserved():
    label = "Manager's approval"
    code, rules = declarative_rules.parse_declare_rules(
        '```python\nrule1 = Init("Manager\'s approval")\n```', [label]
    )
    assert rules == [InitializationRule(label)]
    assert declarative_rules.parse_declare_rules(code, [label])[1] == rules


@pytest.mark.parametrize(
    "source,reason",
    [
        ("import os\nrule1 = Init('A')", "Only DECLARE"),
        ("rule1 = __import__('os')", "Unsupported DECLARE"),
        ("rule1 = os.system('echo unsafe')", "Each statement"),
        ("rule1 = Init(activity)", "literal activity"),
        ("rule1 = Init(f'{1}')", "literal activity"),
        ("rule1 = Init('C')", "Unknown activity"),
        ("rule1 = End('A', 'B')", "Invalid End"),
        ("rule1 = Response('A')", "Invalid Response"),
        ("rule1 = Response('A', 'A')", "distinct activities"),
        ("rule1 = Response('A', 'B', secret='x')", "Invalid arguments"),
        ("rule1 = Init('A')\nrule1 = End('B')", "Duplicate rule variable"),
        ("x = [Init('A')]", "Each statement"),
        ("print('Hello')", "Unsupported DECLARE"),
        ("rule1 = Init('A')\n```", "incomplete"),
        ("", "No DECLARE"),
    ],
)
def test_reject_invalid_or_executable_code(source, reason):
    with pytest.raises(ValueError, match=reason):
        declarative_rules.parse_declare_rules(source, ["A", "B"])


def test_prompt_respects_activity_list():
    prompt = declarative_rules.generate_declare_prompt(
        "Register before completing the case.", ["Register", "Complete"],
        include_examples=False,
    )
    assert "Register before completing the case." in prompt
    assert "'Register', 'Complete'" in prompt
    assert "ChainResponse" in prompt
    assert "Input: Every application" not in prompt
    assert "Input: Every application" in declarative_rules.generate_declare_prompt(
        "Register before completing.", ["Register", "Complete"]
    )


def test_query_uses_existing_llm_retry_flow(monkeypatch):
    recorded = {}

    def fake_retry(conversation, extraction_function, **kwargs):
        recorded.update(kwargs)
        code, rules = extraction_function(
            "```python\nrule1 = Init('A')\nrule2 = Response('A', 'B')\n```", False
        )
        return code, rules, conversation + [
            {"role": "user", "content": "format correction", "type": "error"}
        ]

    monkeypatch.setattr(
        declarative_rules, "generate_result_with_error_handling", fake_retry
    )
    connection = LLMConnection("test-key", "test-model", "test-provider", args={"x": 1})
    rules, corrections, code = declarative_rules.query_llm_for_declare_rules(
        "A happens before B.", ["A", "B"], connection
    )
    assert rules == [InitializationRule("A"), ResponseRule("A", "B")]
    assert corrections == 1
    assert "rule2 = Response" in code
    assert recorded["llm_name"] == "test-model"
    assert recorded["api_key"] == "test-key"
    assert recorded["llm_args"] == {"x": 1}
    assert recorded["max_iterations"] == 3


def test_rules_roundtrip_into_powl_discovery():
    import powl

    from powl.discovery.rule_based.conformance import assert_model_satisfies_rules

    log = [["A", "B"], ["A", "B"], ["A", "C", "B"]]
    _, rules = declarative_rules.parse_declare_rules(
        "rule1 = Init('A')\nrule2 = Response('A', 'B')", ["A", "B", "C"]
    )
    measurements = powl.validate_rules(log, rules)
    assert all(result.support == 1.0 for result in measurements)
    model = powl.discover_with_rules(log, rules, repair_mode="event_level")
    assert_model_satisfies_rules(model, rules)


def test_refine_rules_passes_original_code_and_feedback(monkeypatch):
    captured = {}

    def fake_retry(conversation, extraction_function, **kwargs):
        captured["messages"] = conversation
        code, rules = extraction_function("rule1 = End('B')", False)
        return code, rules, conversation

    monkeypatch.setattr(
        declarative_rules, "generate_result_with_error_handling", fake_retry
    )
    result, corrections, code = declarative_rules.refine_declare_rules(
        "A then B", ["A", "B"], "rule1 = Init('A')", "B must be last",
        LLMConnection("key", "model", "provider"),
    )
    assert result == [EndRule("B")]
    assert corrections == 0
    assert code == "rule1 = End('B')"
    assert "Init('A')" in captured["messages"][1]["content"]
    assert "B must be last" in captured["messages"][2]["content"]

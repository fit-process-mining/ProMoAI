import pytest

from powl.discovery.rule_based import ResponseRule
from powl.discovery.rule_based import conformance

from promoai.model_generation import llm_model_generator


def test_refinement_rejects_rule_violations_without_changing_model(monkeypatch):
    original_model = object()
    candidate = object()
    rules = [ResponseRule("A", "B")]
    generator = llm_model_generator.LLMProcessModelGenerator(
        original_model, [{"role": "user", "content": "initial model"}]
    )
    original_history = list(generator.get_conversation())

    def fake_generate(conversation, **kwargs):
        return "code", candidate, conversation

    def reject_model(model, supplied_rules):
        assert model is candidate
        assert supplied_rules == rules
        raise ValueError("Rule violation")

    monkeypatch.setattr(llm_model_generator, "generate_model", fake_generate)
    monkeypatch.setattr(conformance, "assert_model_satisfies_rules", reject_model)

    with pytest.raises(ValueError, match="Rule violation"):
        generator.update(
            "Please change the model", api_key="key", ai_model="model",
            ai_provider="provider", required_rules=rules,
        )

    assert generator.get_powl() is original_model
    assert generator.get_conversation() == original_history


def test_refinement_keeps_successful_verified_model(monkeypatch):
    candidate = object()
    generator = llm_model_generator.LLMProcessModelGenerator(
        object(), [{"role": "user", "content": "initial model"}]
    )
    checked = []
    monkeypatch.setattr(
        llm_model_generator, "generate_model",
        lambda conversation, **kwargs: ("code", candidate, conversation),
    )
    monkeypatch.setattr(
        conformance,
        "assert_model_satisfies_rules",
        lambda model, rules: checked.append((model, rules)),
    )
    required = [ResponseRule("A", "B")]
    generator.update(
        "Change", api_key="key", ai_model="model", ai_provider="provider",
        required_rules=required,
    )
    assert generator.get_powl() is candidate
    assert checked == [(candidate, required)]
    assert len(generator.get_conversation()) == 2

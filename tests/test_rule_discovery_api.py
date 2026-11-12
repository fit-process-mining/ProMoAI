from powl.discovery.rule_based import PrecedenceRule

from promoai import main


def test_discovery_delegates_to_powl_and_wraps_result(monkeypatch):
    captured = {}
    model = object()
    wrapper = object()

    def fake_discovery(log, rules, **kwargs):
        captured["log"] = log
        captured["rules"] = rules
        captured["kwargs"] = kwargs
        return model

    monkeypatch.setattr(main, "discover_with_rules", fake_discovery)
    monkeypatch.setattr(
        main.LLMProcessModelGenerator,
        "from_powl",
        lambda powl_model: wrapper if powl_model is model else None,
    )
    rules = [PrecedenceRule("A", "B")]
    log = [["A", "B"]]
    result = main.generate_model_from_event_log_with_rules(
        log, rules, repair_mode="edit_distance", noise_threshold=0.25,
        max_states=200, verify_rules=True,
    )
    assert result is wrapper
    assert captured["log"] is log
    assert captured["rules"] is rules
    assert captured["kwargs"] == {
        "repair_mode": "edit_distance", "noise_threshold": 0.25,
        "verify_rules": True, "max_states": 200,
    }


def test_validate_delegates_to_powl(monkeypatch):
    output = object()
    monkeypatch.setattr(main, "validate_rules", lambda *args, **kwargs: output)
    assert main.validate_event_log_rules([["A"]], []) is output


def test_public_prompt_api_uses_llm_credentials(monkeypatch):
    captured = {}

    def fake_generate(description, activities, connection, *, include_examples):
        captured["connection"] = connection
        captured["include_examples"] = include_examples
        return [PrecedenceRule("A", "B")], 0, "rule1 = Precedence('A', 'B')"

    monkeypatch.setattr(main, "query_llm_for_declare_rules", fake_generate)
    result = main.generate_rules_from_text(
        "A before B", ["A", "B"], "api-key", "model", "provider",
        llm_args={"trace": "value"}, include_examples=False,
    )
    assert result == [PrecedenceRule("A", "B")]
    assert captured["connection"].api_key == "api-key"
    assert captured["connection"].llm_name == "model"
    assert captured["connection"].ai_provider == "provider"
    assert captured["connection"].args == {"trace": "value"}
    assert not captured["include_examples"]

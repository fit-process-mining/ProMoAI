import pandas as pd
import pytest

from powl.discovery.rule_based import InitializationRule, ResponseRule
from rule_discovery_page import _rule_label, _validate_log, _validation_log


def test_rule_labels_quote_activities():
    assert _rule_label(InitializationRule("Manager's approval")) == (
        'Init("Manager\'s approval")'
    )
    assert _rule_label(ResponseRule("A", "B")) == "Response('A', 'B')"


def test_columns_and_activity_validation():
    log = pd.DataFrame({
        "case": ["1", "1", "2"],
        "activity": ["B", "A", "B"],
    })
    assert _validate_log(log, "activity", "case") == ["A", "B"]
    with pytest.raises(ValueError, match="must be different"):
        _validate_log(log, "case", "case")
    log.loc[0, "activity"] = None
    with pytest.raises(ValueError, match="missing values"):
        _validate_log(log, "activity", "case")


def test_lifecycle_filter_matches_powl_completion_handling():
    log = pd.DataFrame({
        "case:concept:name": ["1", "1", "2"],
        "concept:name": ["A", "A", "B"],
        "lifecycle:transition": ["start", "complete", "COMPLETE"],
    })
    filtered = _validation_log(log, True)
    assert filtered["concept:name"].tolist() == ["A", "B"]
    assert _validation_log(log, False) is log
    without_completions = log.loc[log["lifecycle:transition"] == "start"]
    assert _validation_log(without_completions, True) is without_completions

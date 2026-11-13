"""Discovery and import must retain process semantics when refined as code."""

from copy import deepcopy

import pytest

import powl
from promoai.model_generation.generator import ModelGenerator
from promoai.model_generation.llm_model_generator import LLMProcessModelGenerator
from promoai.model_generation.safe_model import evaluate_model_code
from promoai.pn_to_powl.converter_utils.powl_to_code import translate_powl_to_code
from promoai.prompting.shots import SHOTS, RESOURCE_AWARE_SHOTS


@pytest.mark.parametrize("shot", SHOTS + RESOURCE_AWARE_SHOTS)
def test_all_builtin_examples_roundtrip_through_safe_model_language(shot):
    original = shot[1]()
    restored = evaluate_model_code(translate_powl_to_code(original))
    assert original.same_structure(restored)


def test_silent_nodes_tags_resources_and_annotations_are_preserved():
    gen = ModelGenerator()
    a = gen.activity("Manager's \"review\"\ncomplete", pool="O'Reilly")
    gen.annotate(a, cost=12.5, description="Check the invoice")
    b = gen.activity("B", pool="O'Reilly")
    original = gen.skip(gen.self_loop(gen.sequence(a, gen.skip(b), gen.silent_transition())))
    before = deepcopy(original)
    restored = evaluate_model_code(translate_powl_to_code(original))
    assert original.same_structure(restored)
    assert original.same_structure(before)


def test_discovered_model_can_be_read_as_code_without_provider_credentials():
    discovered = powl.discover_with_rules([["A", "B"], ["A", "C", "B"]], [])
    wrapper = LLMProcessModelGenerator.from_powl(discovered)
    restored = evaluate_model_code(wrapper.get_code())
    assert discovered.same_structure(restored)


def test_unsupported_frequencies_fail_instead_of_losing_semantics():
    original = powl.Activity("A", min_freq=2, max_freq=3)
    with pytest.raises(ValueError, match="min_freq"):
        translate_powl_to_code(original)

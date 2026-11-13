from pm4py import BPMN, convert_to_petri_net, PetriNet
from powl.discovery.rule_based import discover_with_rules, validate_rules
from powl.main import discover as discover_powl

from promoai.aipa.bpmn_analyzer import BPMNAnalyzer
from promoai.model_generation.llm_model_generator import LLMProcessModelGenerator
from promoai.general_utils.llm_connection import LLMConnection
from promoai.pn_to_powl.converter import convert_workflow_net_to_powl
from promoai.prompting.declarative_rules import (
    query_llm_for_declare_rules,
    refine_declare_rules,
)


def generate_model_from_text(
    description: str,
    api_key: str,
    ai_model: str,
    ai_provider: str,
    resource_aware_discovery: bool = False,
    llm_args: dict = None,
):
    return LLMProcessModelGenerator.from_description(
        description,
        api_key,
        ai_model,
        ai_provider,
        resource_aware_discovery=resource_aware_discovery,
        llm_args=llm_args,
    )


def generate_model_from_event_log(event_log, threshold=0.0):
    powl_model = discover_powl(event_log, dfg_frequency_filtering_threshold=threshold)
    return LLMProcessModelGenerator.from_powl(powl_model=powl_model)


def generate_rules_from_text(
    description: str,
    activities,
    api_key: str,
    ai_model: str,
    ai_provider: str,
    llm_args: dict | None = None,
    include_examples: bool = True,
):
    connection = LLMConnection(api_key, ai_model, ai_provider, args=llm_args)
    rules, _, _ = query_llm_for_declare_rules(
        description, activities, connection, include_examples=include_examples
    )
    return rules


def refine_rules_from_feedback(
    description: str,
    activities,
    previous_code: str,
    feedback: str,
    api_key: str,
    ai_model: str,
    ai_provider: str,
    llm_args: dict | None = None,
):
    connection = LLMConnection(api_key, ai_model, ai_provider, args=llm_args)
    rules, _, _ = refine_declare_rules(
        description, activities, previous_code, feedback, connection
    )
    return rules


def generate_model_from_event_log_with_rules(
    event_log,
    rules,
    repair_mode: str = "event_level",
    noise_threshold: float = 0.0,
    verify_rules: bool = True,
    **discovery_kwargs,
):
    powl_model = discover_with_rules(
        event_log,
        rules,
        repair_mode=repair_mode,
        noise_threshold=noise_threshold,
        verify_rules=verify_rules,
        **discovery_kwargs,
    )
    return LLMProcessModelGenerator.from_powl(powl_model=powl_model)


def validate_event_log_rules(event_log, rules, **validation_kwargs):
    return validate_rules(event_log, rules, **validation_kwargs)


def generate_model_from_petri_net(pn: PetriNet):
    powl_model = convert_workflow_net_to_powl(pn)
    return LLMProcessModelGenerator.from_powl(powl_model=powl_model)


def generate_model_from_bpmn(bpmn_diagram: BPMN):
    pn, im, fm = convert_to_petri_net(bpmn_diagram)
    return generate_model_from_petri_net(pn)


def query_bpmn(
    bpmn_file_path: str, query: str, api_key: str, ai_model: str, ai_provider: str
):
    bpmn_xml_string = open(bpmn_file_path, "r").read()
    return BPMNAnalyzer(
        bpmn_xml_string=bpmn_xml_string,
        initial_query=query,
        api_key=api_key,
        ai_model=ai_model,
        ai_provider=ai_provider,
    )

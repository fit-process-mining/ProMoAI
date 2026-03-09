from pm4py import BPMN, convert_to_petri_net, PetriNet
from powl.main import discover as discover_powl

from promoai.aipa.bpmn_analyzer import BPMNAnalyzer
from promoai.model_generation.llm_model_generator import LLMProcessModelGenerator
from promoai.pn_to_powl.converter import convert_workflow_net_to_powl


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

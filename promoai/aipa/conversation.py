from promoai.aipa.abstraction import get_simplified_xml_abstraction
from promoai.aipa.prompting import add_prompt_strategies


def create_conversation(
    role="system",
    model_abstraction="simplified_xml",
    enable_role_prompting=True,
    enable_natural_language_restriction=True,
    enable_chain_of_thought=False,
    enable_process_analysis=False,
    enable_knowledge_injection=False,
    enable_few_shots_learning=False,
    enable_negative_prompting=False,
    enable_examples=True,
):

    prompt = add_prompt_strategies(
        model_abstraction,
        enable_role_prompting,
        enable_natural_language_restriction,
        enable_chain_of_thought,
        enable_process_analysis,
        enable_knowledge_injection,
        enable_few_shots_learning,
        enable_negative_prompting,
        enable_examples,
    )
    conversation = [
        create_message(prompt, role=role, model_abstraction=model_abstraction)
    ]
    return conversation


def create_message(
    message: str,
    role,
    model_abstraction,
    additional_content=None,
    additional_content_type: str = "text",
):

    if model_abstraction not in ["svg"]:
        content = message
    else:
        content = [{"type": "text", "text": f"{message}"}]
        if additional_content is not None:
            content.append(
                {
                    "type": additional_content_type,
                    additional_content_type: additional_content,
                }
            )

    message = {"role": role, "content": content}

    return message


def create_process_model_representation(
    model_abstraction,
    role: str = "system",
    xml_string: str = None,
    json_abstraction: str = None,
) -> dict:

    if model_abstraction == "json":
        if json_abstraction is None:
            raise ValueError("json_abstraction must be provided")
        message = f"This is a JSON representing the full BPMN model: {json_abstraction}"

    elif model_abstraction == "xml":
        if xml_string is None:
            raise ValueError("xml_string must be provided")
        message = (
            f"This is a text containing the BPMN 2.0 XML of the process: {xml_string}"
        )

    elif model_abstraction == "svg":
        raise ValueError("SVG is not supported!")

    elif model_abstraction == "simplified_xml":
        if xml_string is None:
            raise ValueError("xml_string must be provided")
        simplified_xml_string = get_simplified_xml_abstraction(xml_string)
        message = f"This is an XML-like textual representation of the BPMN: {simplified_xml_string}"

    else:
        raise Exception(f"Unknown model abstraction: {model_abstraction}")

    abstraction_message = create_message(
        message, role=role, model_abstraction=model_abstraction
    )

    return abstraction_message

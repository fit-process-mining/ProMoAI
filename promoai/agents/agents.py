import copy
from functools import partial
from typing import Tuple

from promoai.agents.pm4py_wrapper import PM4PYWrapper
from promoai.agents.state import ProcessState
from promoai.agents.utils import code_extraction_report
from promoai.general_utils.llm_connection import (
    generate_result_with_error_handling,
    LLMConnection,
)


def init_state(user_request: str, event_log) -> ProcessState:
    initial_state = ProcessState(user_request=user_request, event_log=event_log)
    return initial_state


def engineer_node(
    state: ProcessState, LLMCredentials: LLMConnection
) -> Tuple[ProcessState, str, str]:
    # Initial state
    # 1. Construct Prompt
    api = PM4PYWrapper(state)
    api_summary = api.get_API_summary()

    msg = f"""
    You are a Process Mining Data Engineer.

    Your task is to preprocess the event log data according to the user's request, using the provided API methods and any necessary data manipulation with pandas or numpy.
    {api_summary} \n

    DATA: \n
    - Assume that the log is already loaded and accessible via api.event_log. \n
    - Furthermore, the data has the following structure: {state['log_abstraction']} \n
    - Activities are in the 'concept:name' column, timestamps in 'time:timestamp', case identifier in 'case:concept:name'. \n

    EXPECTED OUTPUT:
    - Always return the final event log as `final_event_log` variable after any preprocessing steps, so it can be passed down to the subsequent agent.

    IMPORTANT:
    - If the previous code you generated is able to answer the user's new request without any modifications, just return the event log without any changes.


    """
    msg_history = (
        [
            {"role": "system", "content": msg},
            {"role": "user", "content": state["user_request"][-1]},
        ]
        if len(state["messages_eng"]) == 0
        else state["messages_eng"]
        + [{"role": "user", "content": state["user_request"][-1]}]
    )
    code, result, messages = generate_result_with_error_handling(
        msg_history,
        extraction_function=api.code_extraction,
        llm_name=LLMCredentials.llm_name,
        ai_provider=LLMCredentials.ai_provider,
        api_key=LLMCredentials.api_key,
        llm_args=LLMCredentials.args,
        max_iterations=3,
        additional_iterations=2,
    )
    state["event_log"] = result
    state["log_abstraction"] = state.generate_log_abstraction()
    state["messages_eng"].extend(messages)
    return state, result, code


def update_message_for_analyst(last_request: str, context: str) -> str:
    msg = f"""
    The Process Engineer has executed the following preprocessing steps on the event log based on the user's request "{context}":
    Please use this information to generate a report, making sure to reference artifacts that are relevant to the current user request and to support your analysis with the provided visualizations and data summaries.

    REMEMBER:
    - YOUR GOAL is to answer the User Request: "{last_request}" by presenting and interpreting the provided artifacts. \n
    - Recall the output format: \n
    ```python
    final_report = [
        {{"type": "text", "content": "### Analysis of Process Distribution\\nAs requested, I have analyzed the activity distribution..."}},
        {{"type": "artifact", "content": "artifact_0"}},
        {{"type": "text", "content": "The chart above (artifact_0) clearly shows that..."}}
    ]
    ```
    \n
    - Make sure to follow the strict rules introduced in the initial message (e.g., referencing artifacts by their exact keys, including the visualization if the user asked for it and it is provided, etc.) when generating your report.
    - Do NOT mention any errors or internal processing steps. Your report should be a polished analysis that directly addresses the user's request using the provided artifacts as evidence to support your conclusions.
    You will receive the artifacts in the consequent message.
    """

    return msg


def generate_initial_message_for_analyst(
    state: ProcessState, context, artifact_df_strings, artifact_viz_strings
) -> str:
    msg = f"""
    You are a process analyst.

    CRITICAL INSTRUCTION:
    The Process Engineer has already executed the analysis. Your SOLE TASK is to compile the final report using the artifacts provided below.
    - DO NOT ask the user if they want to generate a visualization; it has ALREADY been generated.
    - DO NOT say "I can generate..." or "If you want me to...".
    - YOUR GOAL is to answer the User Request: "{state['user_request'][-1]}" by presenting and interpreting the provided artifacts.

    DATA SUMMARY:
    - Log Abstraction: {state['log_abstraction']}

    TASK:
    Construct the `final_report` variable. Interleave your analysis with the relevant artifact keys (e.g., "artifact_0").
    - If an artifact is a visualization (PNG), use it to support your findings.
    - If an artifact is a table (CSV), summarize the key insights in text rather than displaying the whole table instead of adding the entire potentially large table to your report unless explicitly asked by the user.
    - For visualizations, you will get an abstraction in form of a description (content of the artifact or its summary). Use it to identify if it is relevant and to support your analysis.

    OUTPUT FORMAT:
    Your output must be Python code that defines a variable `final_report` wrapped in a python code block (```python ... ```).

    Example:
    ```python
    final_report = [
        {{"type": "text", "content": "### Analysis of Process Distribution\\nAs requested, I have analyzed the activity distribution..."}},
        {{"type": "artifact", "content": "artifact_0"}},
        {{"type": "text", "content": "The chart above (artifact_0) clearly shows that..."}}
    ]
    ```

    STRICT RULES:
    1. Reference artifacts by their EXACT keys (e.g., "artifact_0").
    2. Do NOT mention any errors or internal processing steps.
    3. If the user asked for a visualization and it is provided in the ARTIFACTS section, you MUST include it in the report.
    4. You may reuse artifacts from previous iterations by referencing their keys, but you cannot introduce new artifacts that were not provided in the ARTIFACTS section.
    5. Before generating the report, carefully analyze the provided artifacts, respond after making sure you have fully utilized the available information to answer the user's request.
    """
    return msg


def analyst_node(state: ProcessState, LLMCredentials: LLMConnection) -> ProcessState:
    artifact_id_to_description_and_content = {}
    artifact_id_to_filepath = {}
    sent_artifacts = state["sent_artifacts"]

    for i, (file_path, (description, data)) in enumerate(
        state["saved_artifacts"].items()
    ):
        aid = f"artifact_{i}"
        clean_description = str(description).strip("{}'\" ")
        artifact_id_to_filepath[aid] = file_path

        a_type = (
            "visualization"
            if file_path.endswith((".png", ".jpg", ".jpeg"))
            else "dataframe"
            if file_path.endswith(".csv")
            else None
        )

        if a_type is None:
            raise Exception(
                f"Unsupported artifact type for artifact {file_path}. Only image and csv files are supported."
            )

        a_content = (
            "Visual representation available (PNG). Reference this key to display the chart"
            if a_type == "visualization"
            else data
        )

        artifact_id_to_description_and_content[aid] = {
            "description": clean_description,
            "content": a_content,
            "type": a_type,
        }

    artifact_dataframes = {
        k: v
        for k, v in artifact_id_to_description_and_content.items()
        if v["type"] == "dataframe"
    }
    artifact_df_strings = "\n".join(
        [
            f"\n =====DATAFRAME {k}======:\n Description: {v['description']} \n Content: {v['content']} \n \
                                ======END OF {k} ====== \n\n"
            for k, v in artifact_dataframes.items()
            if k not in sent_artifacts
        ]
    )

    artifact_viz = {
        k: v
        for k, v in artifact_id_to_description_and_content.items()
        if v["type"] == "visualization"
    }
    artifact_viz_strings = "\n".join(
        [
            f"\n =====VISUALIZATION {k}======:\n Description: {v['description']} \n Content: {v['content']} \n \n ======END OF {k} ====== \n\n"
            for k, v in artifact_viz.items()
            if k not in sent_artifacts
        ]
    )
    context = (
        " ".join(state["context"])
        if len(state["context"]) > 0
        else "No additional context available."
    )
    msg_history = state["messages_ana"]
    if len(msg_history) == 0:
        initial_msg = generate_initial_message_for_analyst(
            state, context, artifact_df_strings, artifact_viz_strings
        )
        msg_history.append({"role": "system", "content": initial_msg})
        msg_history.append({"role": "user", "content": state["user_request"][-1]})
    else:
        updated_msg = update_message_for_analyst(state["user_request"][-1], context)
        additional_info = f"These are the artifacts that have been generated in the current iteration \
        and are relevant to the user's request but have not been included in the previous report: \n \n \
        {artifact_df_strings} \n\n  \
        {artifact_viz_strings} \n\n \
        Recall the user request: {state['user_request'][-1]} \n\n "
        msg_history.append({"role": "user", "content": updated_msg})
        msg_history.append({"role": "user", "content": additional_info})

    valid_artifact_ids = list(artifact_id_to_description_and_content.keys())
    partial_function = partial(
        code_extraction_report, valid_artifact_ids=valid_artifact_ids
    )
    try:
        result, _, messages = generate_result_with_error_handling(
            msg_history,
            extraction_function=partial_function,
            llm_name=LLMCredentials.llm_name,
            ai_provider=LLMCredentials.ai_provider,
            api_key=LLMCredentials.api_key,
            llm_args=LLMCredentials.args,
            max_iterations=3,
            additional_iterations=2,
        )
        state["sent_artifacts"].extend(
            list(artifact_id_to_description_and_content.keys())
        )

    except Exception as e:
        raise Exception(f"Error during analyst node execution: {e}")

    # Now, we have to postprocess the result to replace the image keys with their actual content (e.g., file paths)
    postprocessed_report = copy.deepcopy(result)
    for entry in postprocessed_report:
        if entry.get("type") == "artifact":
            key = entry.get("content")
            if key in artifact_id_to_filepath:
                entry["content"] = artifact_id_to_filepath[key]
            else:
                raise ValueError(f"Key {key} not found.")
    state["messages_ana"].extend(messages)
    state["final_report"] = postprocessed_report
    state.flush_context()
    return state

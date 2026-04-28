import copy
from functools import partial
from typing import Tuple, Any

import pandas as pd
import pm4py

from promoai.agents.pm4py_wrapper import PM4PYWrapper, LLMClient
from promoai.agents.state import ProcessState
from promoai.agents.utils import code_extraction_report
from promoai.general_utils.artifact_store import (
    append_manifest_entry,
    create_managed_path,
    write_json_artifact,
    write_text_artifact,
)
from promoai.general_utils.llm_connection import (
    generate_result_with_error_handling,
    LLMConnection,
)

ERROR_MESSAGE_CODE_GENERATION_ENG = """
Please update the model to fix the error. Make sure" \
                          f" to save the updated final event log is the variable 'final_event_log'. """

ERROR_MESSAGE_CODE_GENERATION_ANALYST = """
Please update the model to fix the error. Make sure to save the final report in the variable 'final_report' as a list of dictionaries with keys 'type' and 'content', where 'type' can be either 'text' or 'artifact'. For artifacts, the content should be the key referencing the artifact (e.g., 'artifact_0') that you want to include in the report."""


def _persist_generated_code(
    state: ProcessState, role: str, request_index: int, user_request: str, code: str
) -> str:
    file_path = write_text_artifact(
        state["artifact_session_dir"],
        "code",
        f"{role}_step_{request_index}_{user_request}",
        code,
        suffix=".py",
        prefix=role,
    )
    append_manifest_entry(
        state["artifact_session_dir"],
        category="code",
        file_path=file_path,
        description=f"{role.title()} generated code for request {request_index}.",
        artifact_type="generated_code",
        extra={
            "role": role,
            "request_index": request_index,
            "user_request": user_request,
        },
    )
    return file_path


def _persist_event_log_snapshot(
    state: ProcessState, event_log, request_index: int, label: str
) -> str | None:
    try:
        dataframe = (
            event_log
            if isinstance(event_log, pd.DataFrame)
            else pm4py.convert_to_dataframe(event_log)
        )
    except Exception:
        return None

    file_path = create_managed_path(
        state["artifact_session_dir"],
        "event_logs",
        f"request_{request_index}_{label}",
        ".csv",
        prefix="event_log",
    )
    dataframe.to_csv(file_path, index=False)
    append_manifest_entry(
        state["artifact_session_dir"],
        category="event_logs",
        file_path=file_path,
        description=f"Event log snapshot after {label} for request {request_index}.",
        artifact_type="event_log_snapshot",
        extra={"rows": len(dataframe), "columns": list(dataframe.columns)},
    )
    return file_path


def init_state(
    user_request: str,
    event_log,
    process_model : str | None = None,
    artifact_session_dir: str | None = None,
    source_log_path: str | None = None,
) -> ProcessState:
    initial_state = ProcessState(
        user_request=user_request,
        event_log=event_log,
        process_model = process_model,
        artifact_session_dir=artifact_session_dir,
        source_log_path=source_log_path,
    )
    return initial_state


def engineer_node(
    state: ProcessState, LLMCredentials: LLMConnection
) -> Tuple[ProcessState, str, str]:
    # Initial state
    # 1. Construct Prompt
    client = LLMClient(LLMCredentials)
    api = PM4PYWrapper(state, client)
    api_summary = api.get_API_summary()

    # Helpers
    has_log = state["event_log"] is not None and len(state["event_log"])
    has_model = state["process_model"] is not None
    log_context = f"- Log Structure: {state.get('log_abstraction', 'No log loaded.')}. \n The log is accessible via api.event_log" if has_log else "- Log Status: No event log has been provided yet."
    model_context = f"- Initial Model's Structure: {state.get('model_abstraction', '')}. \n"
    print(f"Model context: {model_context}")
    msg = f"""
        You are a Process Mining Specialist and Data Engineer.

        Your task is to handle the user's request by manipulating event data, analyzing process models, or aligning the two.
        {api_summary} \n

        CONTEXT:
        {log_context} \n
        {model_context} \n

        OPERATIONAL MODES:
        1. DATA-DRIVEN: If an event log is present, you can filter, clean, or mine it.
        - MANDATORY: Ensure 'concept:name', 'time:timestamp', and 'case:concept:name' are correctly mapped. 
        2. MODEL-DRIVEN: If a model is present, use it for analysis. 
        - STICKINESS RULE: If a model is already available in the CONTEXT, you MUST use it. 
        - DO NOT call `api.discover_process_model()` if a model already exists unless the user specifically asks to "re-mine" or "generate a new model from data."
        3. COMBINED: Use the existing model and log together (e.g., Conformance Checking) without replacing the existing model.

        RULES:
        - PRECEDENCE: A user-provided process model takes precedence. Always use `api.get_model_summary()` or `api.cc_alignments()` on the existing model rather than discovering a new one.
        !!IMPORTANT!!: Calling discovery will delete the user's uploaded model.
        - VARIABLE MAPPING: The log is at `api.event_log`. The current model is handled internally by the `api` methods and available via `api.process_model`.
        - INITIAL INPUT: The initial event log is available as `api.initial_event_log` and should be used for any operations that require the original log state, e.g., restoring the log to its initial state. \n
        The initial process model (e.g., the provided by user if any) is available as `api.initial_process_model` and should be used for any operations that require the original model state, e.g., comparing the current model with the initial one, restoring the model to its initial state, etc.
        - RELEVANCE: If the previous code or current state already satisfies the request, do not perform redundant operations.

        IMPORTANT:
        - COLUMN MAPPING: If activities, timestamps, or IDs are not in standard columns, you MUST rename/preprocess them using pandas or `api.filter_pandas_query`.
        - DESCRIPTIONS: Every `api.save_visualization` or `api.save_dataframe` call MUST include a meaningful, context-rich description string.
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
    effective_llm_args = dict(LLMCredentials.args or {})
    effective_llm_args["artifact_session_dir"] = state["artifact_session_dir"]
    code, result, messages = generate_result_with_error_handling(
        msg_history,
        extraction_function=api.code_extraction,
        llm_name=LLMCredentials.llm_name,
        ai_provider=LLMCredentials.ai_provider,
        api_key=LLMCredentials.api_key,
        llm_args=effective_llm_args,
        max_iterations=3,
        additional_iterations=2,
        standard_error_message=ERROR_MESSAGE_CODE_GENERATION_ENG,
    )
    request_index = len(state["user_request"])
    _persist_generated_code(
        state, "engineer", request_index, state["user_request"][-1], code
    )
    state["event_log"] = result
    state["log_abstraction"] = state.generate_log_abstraction()
    state["messages_eng"] = messages
    _persist_event_log_snapshot(state, result, request_index, "engineer_output")
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
    - AVOID adding a dataframe/table to the report unless the user EXPLICITLY asks for it. Instead, summarize the key insights from it.
    You will receive the artifacts in the consequent message.
    """

    return msg


def generate_initial_message_for_analyst(state: ProcessState, context) -> str:
    msg = f"""
    You are a process analyst.

    CRITICAL INSTRUCTION:
    The Process Engineer has already executed the analysis. Your SOLE TASK is to compile the final report using the artifacts provided below.
    - DO NOT ask the user if they want to generate a visualization; it has ALREADY been generated.
    - DO NOT say "I can generate..." or "If you want me to...".
    - YOUR GOAL is to answer the User Request: "{state['user_request'][-1]}" by presenting and interpreting the provided artifacts.

    DATA SUMMARY:
    - Log Abstraction: {state['log_abstraction']}
    - Additional Context: {context}

    TASK:
    Construct the `final_report` variable. Interleave your analysis with the relevant artifact keys (e.g., "artifact_0").
    - If an artifact is a visualization (PNG), use it to support your findings.
    - If an artifact is a table (CSV), summarize the key insights in text if it's relevant. DO NOT include the dataframe in the report unless EXPLICITLY stated by user in the request.
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
    6. In the text, DO NOT refer to the artifacts as "artifact_0" but rather as "the chart above", "the chart below", etc. based on the type of artifact and its relevance to the analysis.
    7. DO NOT include dataframes in your report unless explicitly asked, instead, summarize the key insights from the dataframe in text form.
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
    long_dfs = [
        k for k, v in artifact_dataframes.items() if len(str(v["content"])) > 500
    ]
    artifact_df_strings = "\n".join(
        [
            f"\n =====DATAFRAME {k}======:\n Description: {v['description']} \n Content: {v['content']} \n \
                                ======END OF {k} ====== \n\n"
            for k, v in artifact_dataframes.items()
            if k not in sent_artifacts
        ]
    )
    artifact_df_strings += "\n \n IMPORTANT: DATAFRAMES SHOULD NOT BE INCLUDED IN THE REPORT IF THE USER DID NOT REQUEST THEM EXPLICITLY. \n \n"

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
    additional_info = f"These are the artifacts that have been generated in the current iteration \
        and are relevant to the user's request but have not been included in the previous report: \n \n \
        {artifact_df_strings} \n\n  \
        {artifact_viz_strings} \n\n \
        Recall the user request: {state['user_request'][-1]} \n\n "

    if len(msg_history) == 0:
        initial_msg = generate_initial_message_for_analyst(state, context)
        msg_history.append({"role": "system", "content": initial_msg})
    else:
        updated_msg = update_message_for_analyst(state["user_request"][-1], context)
        msg_history.append({"role": "user", "content": updated_msg})

    msg_history.append({"role": "user", "content": additional_info})

    valid_artifact_ids = list(artifact_id_to_description_and_content.keys())
    partial_function = partial(
        code_extraction_report, valid_artifact_ids=valid_artifact_ids, long_dfs=long_dfs
    )
    try:
        effective_llm_args = dict(LLMCredentials.args or {})
        effective_llm_args["artifact_session_dir"] = state["artifact_session_dir"]
        report_code, result, messages = generate_result_with_error_handling(
            msg_history,
            extraction_function=partial_function,
            llm_name=LLMCredentials.llm_name,
            ai_provider=LLMCredentials.ai_provider,
            api_key=LLMCredentials.api_key,
            llm_args=effective_llm_args,
            max_iterations=3,
            additional_iterations=2,
            standard_error_message=ERROR_MESSAGE_CODE_GENERATION_ANALYST,
        )
        state["sent_artifacts"].extend(
            list(artifact_id_to_description_and_content.keys())
        )
        _persist_generated_code(
            state,
            "analyst",
            len(state["user_request"]),
            state["user_request"][-1],
            report_code,
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
    state["messages_ana"] = messages
    state["final_report"] = postprocessed_report
    report_path = write_json_artifact(
        state["artifact_session_dir"],
        "reports",
        f"assistant_report_request_{len(state['user_request'])}",
        postprocessed_report,
        prefix="report",
    )
    append_manifest_entry(
        state["artifact_session_dir"],
        category="reports",
        file_path=report_path,
        description=f"Structured analyst report for request {len(state['user_request'])}.",
        artifact_type="report",
        extra={"entries": len(postprocessed_report)},
    )
    state.flush_context()
    return state

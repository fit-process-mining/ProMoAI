import os
from io import BytesIO
from typing import Dict, List, Union

import markdown
import pandas as pd
import pm4py
import streamlit as st
from constants import MAX_FILE_SIZE
from powl import import_event_log

from promoai.agents.agents import analyst_node, engineer_node, init_state
from promoai.general_utils.artifact_store import (
    append_manifest_entry,
    ARTIFACTS_ROOT,
    create_analysis_session,
    create_managed_path,
    disk_cleanup,
    write_bytes_artifact,
    write_text_artifact,
)
from promoai.general_utils.constants import ENABLE_PATH_EXPOSURE
from promoai.general_utils.llm_connection import LLMConnection
from xhtml2pdf import pisa


def get_active_artifact_session_dir() -> str | None:
    agent_state = st.session_state.get("agent_state")
    if agent_state and agent_state.get("artifact_session_dir"):
        return agent_state["artifact_session_dir"]
    return st.session_state.get("artifact_session_dir")

import os
import pm4py
from promoai.general_utils.artifact_store import (
    create_analysis_session,
    write_bytes_artifact,
    append_manifest_entry,
    create_managed_path,
)

def persist_uploaded_process_model(uploaded_file) -> tuple[str, str, tuple]:
    artifact_session_dir = create_analysis_session("pmax")
    file_bytes = uploaded_file.read()
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    raw_model_path = write_bytes_artifact(
        artifact_session_dir,
        "inputs",
        uploaded_file.name,
        file_bytes,
        description=f"uploaded_model_{uploaded_file.name}",
        prefix="model_raw",
    )
    
    append_manifest_entry(
        artifact_session_dir,
        category="inputs",
        file_path=raw_model_path,
        description="Uploaded raw process model file.",
        artifact_type="uploaded_process_model",
        extra={"original_name": uploaded_file.name, "size_bytes": len(file_bytes)},
    )

    if file_extension == ".pnml":
        net, im, fm = pm4py.read_pnml(raw_model_path)
    elif file_extension == ".bpmn":
        bpmn_graph = pm4py.read_bpmn(raw_model_path)
        net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)
    else:
        try:
            net, im, fm = pm4py.read_pnml(raw_model_path)
        except:
            raise ValueError(f"Unsupported model format: {file_extension}. Please upload .pnml or .bpmn")

    normalized_model_path = create_managed_path(
        artifact_session_dir,
        "models",
        "uploaded_model_normalized",
        ".pnml",
        prefix="model",
    )
    
    pm4py.write_pnml(net, im, fm, normalized_model_path)
    
    model_summary = pm4py.llm.abstract_petri_net(net, im, fm)

    append_manifest_entry(
        artifact_session_dir,
        category="models",
        file_path=normalized_model_path,
        description="Normalized Petri net snapshot.",
        artifact_type="model_snapshot",
        extra={
            "format": "Petri Net",
            "places": len(net.places),
            "transitions": len(net.transitions),
            "summary": model_summary
        },
    )

    return artifact_session_dir, raw_model_path, (net, im, fm)

def persist_uploaded_event_log(uploaded_file) -> tuple[str, str, pd.DataFrame]:
    artifact_session_dir = create_analysis_session("pmax")
    file_bytes = uploaded_file.read()
    raw_log_path = write_bytes_artifact(
        artifact_session_dir,
        "inputs",
        uploaded_file.name,
        file_bytes,
        description=f"uploaded_{uploaded_file.name}",
        prefix="event_log_raw",
    )
    append_manifest_entry(
        artifact_session_dir,
        category="inputs",
        file_path=raw_log_path,
        description="Uploaded event log file.",
        artifact_type="uploaded_event_log",
        extra={"original_name": uploaded_file.name, "size_bytes": len(file_bytes)},
    )

    log = import_event_log(raw_log_path)
    normalized_log = (
        log if isinstance(log, pd.DataFrame) else pm4py.convert_to_dataframe(log)
    )
    normalized_log_path = create_managed_path(
        artifact_session_dir,
        "event_logs",
        "uploaded_event_log_normalized",
        ".csv",
        prefix="event_log",
    )
    normalized_log.to_csv(normalized_log_path, index=False)
    append_manifest_entry(
        artifact_session_dir,
        category="event_logs",
        file_path=normalized_log_path,
        description="Normalized event log snapshot right after import.",
        artifact_type="event_log_snapshot",
        extra={"rows": len(normalized_log), "columns": list(normalized_log.columns)},
    )
    return artifact_session_dir, raw_log_path, normalized_log


def persist_pdf_report(pdf_bytes: bytes) -> str | None:
    artifact_session_dir = get_active_artifact_session_dir()
    if not artifact_session_dir:
        return None

    pdf_path = write_bytes_artifact(
        artifact_session_dir,
        "reports",
        "pmax_analysis_report.pdf",
        pdf_bytes,
        description="pmax_analysis_report",
        prefix="pdf_report",
    )
    append_manifest_entry(
        artifact_session_dir,
        category="reports",
        file_path=pdf_path,
        description="PDF export of the PMAx chat report.",
        artifact_type="pdf_report",
        extra={"size_bytes": len(pdf_bytes)},
    )
    return pdf_path


def export_messages_to_pdf(messages) -> bytes:
    html_content = f"""
    <html>
    <head>
        <style>
            @page {{ size: a4; margin: 1cm; }}
            body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.5; }}
            .user {{ color: #2e77d0; font-weight: bold; margin-top: 15px; border-bottom: 1px solid #2e77d0; }}
            .assistant {{ color: #e63946; font-weight: bold; margin-top: 15px; border-bottom: 1px solid #e63946; }}
            .text {{ margin-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 4px; text-align: left; font-size: 8pt; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">PMAx Analysis Report</h1>
    """

    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "PMAx"
        html_content += f"<div class='{msg['role']}'>{role_label}</div>"

        blocks = msg["content"]
        if not isinstance(blocks, list):
            blocks = [{"type": "text", "content": blocks}]

        for block in blocks:
            b_type = block.get("type")
            b_val = block.get("content")

            if b_type == "text":
                # Use markdown library for cleaner HTML (faster for the PDF engine)
                clean_text = markdown.markdown(b_val)
                html_content += f"<div class='text'>{clean_text}</div>"

            elif b_type == "artifact":
                if os.path.exists(b_val):
                    if b_val.lower().endswith((".png", ".jpg", ".jpeg")):
                        # INSTEAD OF BASE64: Use absolute file path
                        abs_path = os.path.abspath(b_val)
                        html_content += (
                            f"<div class='text'><img src='{abs_path}'></div>"
                        )

                    elif b_val.lower().endswith(".csv"):
                        df = pd.read_csv(b_val)
                        # LIMIT ROWS: PDF tables shouldn't be 1000 rows long
                        if len(df) > 30:
                            html_content += (
                                "<p><i>(Showing first 30 rows of data)</i></p>"
                            )
                            df = df.head(30)
                        html_content += df.to_html(index=False)

    html_content += "</body></html>"

    pdf_buffer = BytesIO()
    # xhtml2pdf works better with file paths if we provide a path helper
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)

    return None if pisa_status.err else pdf_buffer.getvalue()


def display_chat_message(role: str, content: Union[str, List[Dict[str, str]]]):
    """Renders either a simple string or structured chat blocks."""
    with st.chat_message(role):
        if isinstance(content, list):
            for block in content:
                b_type = block.get("type")
                b_val = block.get("content")

                if b_type == "text":
                    st.markdown(b_val)

                elif b_type == "code":
                    label = block.get("label", "Generated code")
                    language = block.get("language", "python")
                    expanded = block.get("expanded", False)
                    with st.expander(label, expanded=expanded):
                        st.code(b_val, language=language)

                elif b_type == "artifact":
                    st.markdown("**Artifact:**")
                    if not os.path.exists(b_val):
                        st.error(f"File missing: {b_val}")
                        continue

                    if b_val.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                        st.image(b_val, width="stretch")
                    elif b_val.lower().endswith(".csv"):
                        df = pd.read_csv(b_val)
                        if len(df) > 30:
                            st.markdown(
                                "<p><i>(Showing first 30 rows of data)</i></p>",
                                unsafe_allow_html=True,
                            )
                            df = df.head(30)
                        st.dataframe(df, use_container_width=True)
                    elif b_val.lower().endswith(".parquet"):
                        df = pd.read_parquet(b_val)
                        if len(df) > 30:
                            st.markdown(
                                "<p><i>(Showing first 30 rows of data)</i></p>",
                                unsafe_allow_html=True,
                            )
                            df = df.head(30)
                        st.dataframe(df, use_container_width=True)
        else:
            st.markdown(content)


def chat(llm_credentials: LLMConnection):
    # delete old artifacts from previous sessions to save disk space
    disk_cleanup(ARTIFACTS_ROOT, ttl=1)
    if "pdf_bytes" not in st.session_state:
        st.session_state["pdf_bytes"] = None
    if "pdf_signature" not in st.session_state:
        st.session_state["pdf_signature"] = 0

    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    if prompt := st.chat_input("Enter your message here..."):
        artifact_session_dir = get_active_artifact_session_dir()
        if artifact_session_dir:
            prompt_path = write_text_artifact(
                artifact_session_dir,
                "requests",
                f"user_request_{len(st.session_state.messages)}",
                prompt,
                suffix=".md",
                prefix="request",
            )
            append_manifest_entry(
                artifact_session_dir,
                category="requests",
                file_path=prompt_path,
                description="User request submitted to PMAx.",
                artifact_type="user_request",
            )
        display_chat_message("user", prompt)
        resettable = "messages" in st.session_state and len(
            st.session_state["messages"]
        )
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add user message to chat history
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = init_state(
                user_request=prompt,
                event_log=st.session_state["uploaded_log"],
                process_model=st.session_state["uploaded_model"] if "uploaded_model" in st.session_state else None,
                artifact_session_dir=artifact_session_dir,
                source_log_path=st.session_state.get("uploaded_log_path"),
            )
        else:
            # Add the user request to the agent state
            st.session_state.agent_state["user_request"].append(prompt)
        assistant_placeholder = st.empty()

        with st.spinner("Processing your request..."):
            try:
                st.session_state.agent_state, _, code = engineer_node(
                    st.session_state.agent_state, llm_credentials
                )
            except Exception as e:
                st.error(f"Error during calling the Engineer: {e}")
                return

            assistant_content = [
                {
                    "type": "code",
                    "content": code,
                    "language": "python",
                    "label": "Generated code",
                    "expanded": False,
                },
                {
                    "type": "text",
                    "content": "_Generating report..._",
                },
            ]

            with assistant_placeholder.container():
                display_chat_message("assistant", assistant_content)

            try:
                updated_state = analyst_node(
                    st.session_state.agent_state, llm_credentials
                )
            except Exception as e:
                st.error(f"Error during calling the Analyst: {e}")
                return

            st.session_state.agent_state = updated_state

        report = st.session_state.agent_state["final_report"]

        if isinstance(report, list):
            assistant_content = [
                {
                    "type": "code",
                    "content": code,
                    "language": "python",
                    "label": "Generated code",
                    "expanded": False,
                },
                *report,
            ]
        else:
            assistant_content = [
                {
                    "type": "code",
                    "content": code,
                    "language": "python",
                    "label": "Generated code",
                    "expanded": False,
                },
                {
                    "type": "text",
                    "content": report,
                },
            ]

        # Replace the same live bubble with the final content
        with assistant_placeholder.container():
            display_chat_message("assistant", assistant_content)

        # Persist only the final assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )
        resettable = "messages" in st.session_state and len(
            st.session_state["messages"]
        )
        if st.session_state["resettable"] != resettable:
            st.session_state["resettable"] = resettable
            st.rerun()


def run_page():
    st.title("🤖 PMAx")

    st.subheader("Your Process Mining Expert Agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hey! I am PMAx. How can I help you today?",
            }
        ]

    if "setup_complete" not in st.session_state:
        st.session_state["setup_complete"] = False

    artifact_session_dir = get_active_artifact_session_dir()
    if artifact_session_dir and ENABLE_PATH_EXPOSURE:
        st.sidebar.caption("Artifact folder")
        st.sidebar.code(artifact_session_dir, language=None)
        st.sidebar.caption(
            "The manifest is stored as `manifest.jsonl` inside this folder."
        )

    if not st.session_state["setup_complete"]:

        with st.form(key="model_gen_form"):
            st.markdown("### 🛠️ Agents Setup")
            uploaded_files = st.file_uploader(
                "For **using an agent**, upload an event log. Additionally, you may also upload a process model (in PNML format). \n",
                type=["xes", "gz", "csv", "pnml"],
                accept_multiple_files=True,
                max_upload_size=MAX_FILE_SIZE,
            )
            submission_button = st.form_submit_button(label="Start Analysis")
            if submission_button:
                if "llm_credentials" not in st.session_state:
                    st.error(
                        body="Please complete the setup on the main page!", icon="⚠️"
                    )
                    return
                if uploaded_files is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                try:
                    log_files = [f for f in uploaded_files if f.name.lower().endswith((".xes", ".gz", ".csv"))]
                    if len(log_files) != 1:
                        st.error("Please upload exactly one event log.")
                        return
                    model_files = [f for f in uploaded_files if f.name.lower().endswith(".pnml")]
                    if len(model_files) > 1:
                        st.error("Please upload at most one model.")
                        return
                    uploaded_log = log_files[0] if log_files else None
                    model = model_files[0] if model_files else None
                    with st.spinner("Uploading and processing the files..."):
                        (
                            artifact_session_dir,
                            raw_log_path,
                            log,
                        ) = persist_uploaded_event_log(uploaded_log)
                        if "uploaded_log" not in st.session_state:
                            st.session_state["uploaded_log"] = log
                            st.session_state["uploaded_log_path"] = raw_log_path
                            st.session_state[
                                "artifact_session_dir"
                            ] = artifact_session_dir
                            st.session_state["setup_complete"] = True
                    if model:
                        with st.spinner("Uploading and processing the files..."):
                            (
                                artifact_session_dir,
                                raw_model_path,
                                model,
                            ) = persist_uploaded_process_model(model)
                            if "uploaded_model" not in st.session_state:
                                st.session_state["uploaded_model"] = model
                                st.session_state["uploaded_model_path"] = raw_model_path
                                st.session_state[
                                    "artifact_session_dir"
                                ] = artifact_session_dir
                                st.session_state["setup_complete"] = True
                    st.rerun()
                                
                except Exception as e:
                    st.error(body=f"Error : {e}", icon="⚠️")
                    return

    else:
        if "llm_credentials" not in st.session_state:
            st.error(
                body="LLM credentials are missing! Please go back to the settings and enter your credentials.",
                icon="⚠️",
            )
            return
        chat(st.session_state["llm_credentials"])


if __name__ in {"__main__", "__page__"}:
    os.environ["MPLBACKEND"] = "Agg"
    run_page()

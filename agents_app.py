import streamlit as st
import os

import shutil
import subprocess
import tempfile

import promoai
import streamlit as st
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from promoai.general_utils.llm_connection import LLMConnection
from powl import import_event_log
from typing import List, Dict, Union
import pandas as pd
import numpy as np
from promoai.general_utils.ai_providers import (
    AI_HELP_DEFAULTS,
    AI_MODEL_DEFAULTS,
    DEFAULT_AI_PROVIDER,
    MAIN_HELP,
)
from promoai.agents.agents import  init_state, engineer_node, analyst_node
from promoai.general_utils.app_utils import DISCOVERY_HELP, InputType

def display_chat_message(role: str, content: Union[str, List[Dict[str, str]]]):
    """Renders either a simple string or a structured artifact report."""
    with st.chat_message(role):
        if isinstance(content, list):
            # Loop through the list of blocks
            for block in content:
                b_type = block.get("type")
                b_val = block.get("content")

                if b_type == "text":
                    st.markdown(b_val)
                
                elif b_type == "artifact":
                    if not os.path.exists(b_val):
                        st.error(f"File missing: {b_val}")
                        continue
                    
                    # Logic for Images vs Dataframes
                    if b_val.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        st.image(b_val, use_container_width=True)
                    elif b_val.lower().endswith('.csv'):
                        df = pd.read_csv(b_val)
                        st.dataframe(df, use_container_width=True)
                    elif b_val.lower().endswith('.parquet'):
                        df = pd.read_parquet(b_val)
                        st.dataframe(df, use_container_width=True)
        else:
            st.markdown(content)

def chat(llm_credentials: LLMConnection):
    
    for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
    
    if prompt:= st.chat_input("Enter your message here..."):
        display_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add user message to chat history
        if "agent_state" not in st.session_state:
            st.session_state.agent_state = init_state(user_request=prompt, event_log=st.session_state['uploaded_log'])
        else:
            # Add the user request to the agent state
            st.session_state.agent_state['user_request'].append(prompt)
        with st.status("I am processing your request...", state = "running", expanded = True) as status:
            status.update(label = "The engineer is analyzing the request and preparing code...", state="running", expanded = True)
            st.session_state.agent_state, _, code = engineer_node(st.session_state.agent_state, llm_credentials)
            st.write("Code generated! Executing the code and generating the report...")
            st.write(f"Executing the following code:\n```python\n{code}\n```")
            # Add the generated code to the status
            status.update(label = "The analyst is generating the report based on the generated code and the analysis artifacts...", state="running")
            updated_state = analyst_node(st.session_state.agent_state, llm_credentials)
            st.session_state.agent_state = updated_state
            status.update(label="Report Generated! ✅", state="complete", expanded = False)




        report = st.session_state.agent_state['final_report']
        display_chat_message("assistant", report)
        st.session_state.messages.append({"role": "assistant", "content": report})
    


def run_app():
    st.title("🤖 ProMoAgents")

    st.subheader("Process Modeling with Generative AI")

    temp_dir = "temp"

    os.makedirs(temp_dir, exist_ok=True)

    if "provider" not in st.session_state:
        st.session_state["provider"] = DEFAULT_AI_PROVIDER

    if "model_name" not in st.session_state:
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def update_model_name():
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]
    if "setup_complete" not in st.session_state:
        st.session_state["setup_complete"] = False

    if st.session_state["setup_complete"]:
        if st.sidebar.button("🔄 Change Settings"):
            st.session_state["setup_complete"] = False
            del st.session_state["uploaded_log"]
            del st.session_state["agent_state"]
            del st.session_state["messages"]
            st.rerun()
        

    if not st.session_state["setup_complete"]:
        st.header("🔧 Agents Setup")
        with st.expander("🔒 Credentials", expanded=True):
            provider = st.selectbox(
                "Choose AI Provider:",
                options=AI_MODEL_DEFAULTS.keys(),
                index=0,
                help=MAIN_HELP,
                on_change=update_model_name,
                key="provider",
            )

            if (
                "model_name" not in st.session_state
                or st.session_state["provider"] != provider
            ):
                st.session_state["model_name"] = AI_MODEL_DEFAULTS[provider]

            col1, col2 = st.columns(2)
            with col1:
                ai_model_name = st.text_input(
                    "Enter the AI model name:",
                    key="model_name",
                    help=AI_HELP_DEFAULTS[st.session_state["provider"]],
                )
            with col2:
                api_key = st.text_input("API key:", type="password")
                st.session_state["llm_credentials"] = LLMConnection(
                    api_key=api_key,
                    llm_name=ai_model_name,
                    ai_provider=provider
                )            
            

        input_type = InputType.DATA.value

        with st.form(key="model_gen_form"):
            uploaded_log = st.file_uploader(
                "For **using an agent**, upload an event log:",
                type=["xes", "gz"],
                help=DISCOVERY_HELP,
            )
            submission_button = st.form_submit_button(label="Start Analysis 🚀")
            if submission_button:
                if not api_key:
                    st.error(body="Please enter your API key!", icon="⚠️")
                    return
                if uploaded_log is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                try:
                    with st.spinner("Uploading and processing the event log..."):
                        contents = uploaded_log.read()
                        os.makedirs(temp_dir, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            mode="wb", delete=False, dir=temp_dir, suffix=uploaded_log.name
                        ) as temp_file:
                            temp_file.write(contents)
                            log = import_event_log(temp_file.name)
                            if "uploaded_log" not in st.session_state:
                                st.session_state["uploaded_log"] = log
                                st.session_state["setup_complete"] = True

                                st.rerun()
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.error(body=f"Error : {e}", icon="⚠️")
                    return
            
    else:
        if "llm_credentials" not in st.session_state:
            st.error(body="LLM credentials are missing! Please go back to the settings and enter your credentials.", icon="⚠️")
            return
        chat(st.session_state["llm_credentials"])


if __name__ == "__main__":
    os.environ['MPLBACKEND'] = 'Agg'

    st.set_page_config(page_title="ProMoAgents", page_icon="🤖")
    # footer()
    run_app()


                    
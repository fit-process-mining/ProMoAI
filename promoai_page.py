import os
import subprocess

import promoai
import streamlit as st
from constants import MAX_FILE_SIZE
from pm4py import read_bpmn, read_pnml

from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from powl import convert_to_bpmn, import_event_log
from powl.conversion.variants.to_petri_net import apply as convert_to_petri_net
from promoai.general_utils.app_utils import InputType
from promoai.general_utils.upload_utils import write_uploaded_file_to_staging
from rule_discovery_page import render_rule_based_discovery


def get_model_svg(model_obj, view_type):
    image_format = "svg"
    powl = model_obj.get_powl()

    if view_type == "POWL":
        from powl.visualization.powl import visualizer

        vis = visualizer.apply(powl)
        return vis  # usually returns the source or pipe

    elif view_type == "Petri Net":
        pn, im, fm = convert_to_petri_net(powl)
        visualization = pn_visualizer.apply(
            pn, im, fm, parameters={"format": image_format}
        )
        return visualization.pipe(format="svg").decode("utf-8")

    else:  # BPMN
        bpmn = convert_to_bpmn(powl)
        from pm4py.objects.bpmn.layout import layouter

        layouted_bpmn = layouter.apply(bpmn)
        visualization = bpmn_visualizer.apply(
            layouted_bpmn, parameters={"format": image_format}
        )
        return visualization.pipe(format="svg").decode("utf-8")


def run_model_generator_app():
    subprocess.run(["streamlit", "run", __file__])


def render_submission_form(input_type):
    description, uploaded_log, uploaded_file, threshold = None, None, None, None
    submit_label = "Run"

    if input_type == InputType.TEXT.value:
        st.markdown("### 📝 Text-to-Model")
        st.caption(
            "Describe your process in natural language to generate a process model."
        )

        description = st.text_area(
            "Process Description:",
            placeholder="e.g., 'An employee submits an expense report. The manager reviews it...'",
            height=200,
        )

        submit_label = "Generate"

    elif input_type == InputType.MODEL.value:
        st.markdown("### 🏗️ Model Improvement")
        st.caption("Upload an existing BPMN or Petri Net to further refine it.")

        uploaded_file = st.file_uploader(
            "Upload Model File",
            type=["bpmn", "pnml"],
            key="model_uploaded_file",
            max_upload_size=MAX_FILE_SIZE,
        )
        submit_label = "Analyze"

    elif input_type == InputType.DATA.value:

        st.markdown("### 📊 Data Discovery")
        st.caption("Transform event logs into process models using POWL Miner.")

        uploaded_log = st.file_uploader(
            "Upload Event Log",
            type=["xes", "gz", "csv"],
            key="data_uploaded_log",
            max_upload_size=MAX_FILE_SIZE,
        )

        # Use columns inside the form for settings
        threshold = st.slider(
            "Noise Filtering Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("data_threshold", 0.0),
            step=0.01,
            key="data_threshold",
            help="Higher values filter out more noise and usually produce a simpler model.",
        )

        submit_label = "Discover"
    submit_button = st.form_submit_button(label=submit_label)
    if submit_button and input_type == InputType.TEXT.value:
        if "llm_credentials" not in st.session_state:
            st.error(body="Please complete the setup on the main page!", icon="⚠️")
            return
        try:
            process_model = promoai.generate_model_from_text(
                description,
                api_key=st.session_state["llm_credentials"].api_key,
                ai_model=st.session_state["llm_credentials"].llm_name,
                ai_provider=st.session_state["llm_credentials"].ai_provider,
                llm_args=st.session_state["llm_credentials"].args,
            )

            st.session_state["model_gen"] = process_model
            st.session_state["feedback"] = []
            st.rerun()

        except Exception as e:
            st.error(body=str(e), icon="⚠️")
            return

    if submit_button and input_type == InputType.DATA.value:
        if uploaded_log is None:
            print("Yeahhh")
            st.error(body="No file is selected!", icon="⚠️")
            return
        temp_path = None
        try:
            temp_path = write_uploaded_file_to_staging(uploaded_log)
            log = import_event_log(temp_path)

            process_model = promoai.generate_model_from_event_log(log, threshold)

            st.session_state["model_gen"] = process_model
            st.session_state["feedback"] = []
            st.rerun()

        except Exception as e:
            st.error(body=f"Error during discovery: {e}", icon="⚠️")
            return
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    elif input_type == InputType.MODEL.value:
        if submit_button:
            if uploaded_file is None:
                st.error(body="No file is selected!", icon="⚠️")
                return
            else:
                temp_path = None
                try:
                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    if file_extension == "bpmn":
                        temp_path = write_uploaded_file_to_staging(
                            uploaded_file, suffix=".bpmn"
                        )
                        bpmn_graph = read_bpmn(temp_path)
                        process_model = promoai.generate_model_from_bpmn(bpmn_graph)

                    elif file_extension == "pnml":
                        temp_path = write_uploaded_file_to_staging(
                            uploaded_file, suffix=".pnml"
                        )
                        pn, im, fm = read_pnml(temp_path)
                        process_model = promoai.generate_model_from_petri_net(pn)

                    else:
                        st.error(
                            body=f"Unsupported file format {file_extension}!",
                            icon="⚠️",
                        )
                        return

                    st.session_state["model_gen"] = process_model
                    st.session_state["feedback"] = []
                except Exception as e:
                    st.error(
                        body=f"Please upload a semi-block-structured model! The error message: {e}",
                        icon="⚠️",
                    )
                    return
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                        st.rerun()


def run_page():
    st.title("🤖 ProMoAI")
    st.markdown("##### *Process Modeling with Generative AI*")
    st.divider()

    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "Model"
    has_model = "model_gen" in st.session_state and st.session_state["model_gen"]

    input_type = st.segmented_control(
        "Select Input Type:",
        # options=["Text", "Model", "Data", "Rules"],
        options=["Text", "Model", "Data"],
        selection_mode="single",
        default="Text",
    )

    if input_type != st.session_state["selected_mode"]:
        st.session_state["selected_mode"] = input_type
        st.session_state["model_gen"] = None
        st.session_state["feedback"] = []
        st.rerun()
    if input_type == InputType.RULES.value:
        render_rule_based_discovery()
    elif not has_model:
        with st.form(key="model_gen_form"):
            render_submission_form(input_type)
    else:
        with st.expander("Input Settings", expanded=False):
            with st.form("model_input_form"):
                render_submission_form(input_type)

    if "model_gen" in st.session_state and st.session_state["model_gen"]:
        if st.session_state.get("just_updated"):
            st.toast("Model updated!", icon="✨")
            st.session_state["just_updated"] = False

        # Model view/export
        view_col, export_col = st.columns([0.75, 0.25])

        with view_col:
            tab_bpmn, tab_petri, tab_powl = st.tabs(
                ["📊 BPMN View", "🕸️ Petri Net", "📐 POWL Structure"]
            )

            with tab_bpmn:
                svg_data = get_model_svg(st.session_state["model_gen"], "BPMN")
                st.image(svg_data, use_container_width=True)

            with tab_petri:
                svg_data = get_model_svg(st.session_state["model_gen"], "Petri Net")
                st.image(svg_data, use_container_width=True)

            with tab_powl:
                svg_data = get_model_svg(st.session_state["model_gen"], "POWL")
                st.image(svg_data, use_container_width=True)

        with export_col:
            st.markdown("### 📥 Export")
            with st.container(border=True):
                proc_model = st.session_state["model_gen"]
                powl_obj = proc_model.get_powl()

                bpmn_xml = get_xml_string(convert_to_bpmn(powl_obj))
                st.download_button(
                    "Download BPMN",
                    data=bpmn_xml,
                    file_name="model.bpmn",
                    use_container_width=True,
                )

                pn, im, fm = convert_to_petri_net(powl_obj)
                pnml_xml = export_petri_as_string(pn, im, fm)
                st.download_button(
                    "Download PNML",
                    data=pnml_xml,
                    file_name="model.pnml",
                    use_container_width=True,
                )
        # Human-in-the-loop refinement
        st.markdown("### 🛠️ Refine Model")
        with st.container(border=True):
            feedback_input = st.text_area(
                "What should be changed?",
                placeholder="e.g. 'Add an approval step after payment'",
                label_visibility="collapsed",
                key="feedback_input",
            )

            if st.button("Refine Model", use_container_width=False, type="primary"):
                if "llm_credentials" not in st.session_state:
                    st.error("Setup credentials first!")
                elif not feedback_input.strip():
                    st.warning("Please enter a refinement instruction.")
                else:
                    with st.spinner("Re-modeling..."):
                        process_model = st.session_state["model_gen"]
                        try:
                            process_model.update(
                                feedback_input,
                                api_key=st.session_state["llm_credentials"].api_key,
                                ai_model=st.session_state["llm_credentials"].llm_name,
                                ai_provider=st.session_state["llm_credentials"].ai_provider,
                                llm_args=st.session_state["llm_credentials"].args,
                                required_rules=(
                                    st.session_state.get("rules_enforced")
                                    if input_type == InputType.RULES.value
                                    else None
                                ),
                            )
                        except Exception as exc:
                            st.error(f"Model refinement failed: {exc}")
                        else:
                            st.session_state["feedback"].append(feedback_input)
                            st.session_state["just_updated"] = True
                            st.rerun()

        # History
        if st.session_state["feedback"]:
            st.markdown("### 📜 Revision History")
            with st.container(border=True):
                for idx, msg in enumerate(reversed(st.session_state["feedback"])):
                    with st.expander(
                        f"v{len(st.session_state['feedback']) - idx}: {msg[:40]}...",
                        expanded=(idx == 0),
                    ):
                        st.markdown(f"**Instruction:** {msg}")


if __name__ in {"__main__", "__page__"}:
    run_page()

import os
import shutil
import subprocess
import tempfile

import promoai
import streamlit as st
from pm4py import read_bpmn, read_pnml

from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.util import constants
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from powl import convert_to_bpmn, import_event_log
from powl.conversion.variants.to_petri_net import apply as convert_to_petri_net
from promoai.general_utils.ai_providers import (
    AI_HELP_DEFAULTS,
    AI_MODEL_DEFAULTS,
    DEFAULT_AI_PROVIDER,
    MAIN_HELP,
)
from promoai.general_utils.app_utils import DISCOVERY_HELP, InputType, ViewType


def run_model_generator_app():
    subprocess.run(["streamlit", "run", __file__])


def run_app():
    st.title("🤖 ProMoAI")

    st.subheader("Process Modeling with Generative AI")

    temp_dir = "temp"

    if "provider" not in st.session_state:
        st.session_state["provider"] = DEFAULT_AI_PROVIDER

    if "model_name" not in st.session_state:
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]

    def update_model_name():
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]

    with st.expander("🔧 Configuration", expanded=True):
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

    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "Model Generation"

    input_type = st.radio(
        "Select Input Type:",
        options=[InputType.TEXT.value, InputType.MODEL.value, InputType.DATA.value],
        horizontal=True,
    )

    if input_type != st.session_state["selected_mode"]:
        st.session_state["selected_mode"] = input_type
        st.session_state["model_gen"] = None
        st.session_state["feedback"] = []
        st.rerun()

    with st.form(key="model_gen_form"):
        if input_type == InputType.TEXT.value:
            description = st.text_area(
                "For **process modeling**, enter the process description:"
            )

            submit_button = st.form_submit_button(label="Run")
            if submit_button:
                try:
                    process_model = promoai.generate_model_from_text(
                        description,
                        api_key=api_key,
                        ai_model=ai_model_name,
                        ai_provider=provider,
                    )

                    st.session_state["model_gen"] = process_model
                    st.session_state["feedback"] = []

                except Exception as e:
                    st.error(body=str(e), icon="⚠️")
                    return
        elif input_type == InputType.DATA.value:
            uploaded_log = st.file_uploader(
                "For **process model discovery**, upload an event log:",
                type=["xes", "gz"],
                help=DISCOVERY_HELP,
            )

            threshold = st.number_input(
                label="Noise Filtering Threshold (0.0 = No Filtering)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

            submit_button = st.form_submit_button(label="Run")
            if submit_button:
                if uploaded_log is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                try:
                    contents = uploaded_log.read()
                    os.makedirs(temp_dir, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, dir=temp_dir, suffix=uploaded_log.name
                    ) as temp_file:
                        temp_file.write(contents)
                        log = import_event_log(temp_file.name)
                    shutil.rmtree(temp_dir, ignore_errors=True)

                    process_model = promoai.generate_model_from_event_log(
                        log, threshold
                    )

                    st.session_state["model_gen"] = process_model
                    st.session_state["feedback"] = []
                except Exception as e:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.error(body=f"Error during discovery: {e}", icon="⚠️")
                    return
        elif input_type == InputType.MODEL.value:
            uploaded_file = st.file_uploader(
                "For **process model improvement**, upload a semi-block-structured BPMN or Petri net:",
                type=["bpmn", "pnml"],
            )
            submit_button = st.form_submit_button(label="Upload")
            if submit_button:
                if uploaded_file is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                else:
                    try:
                        file_extension = uploaded_file.name.split(".")[-1].lower()

                        if file_extension == "bpmn":
                            contents = uploaded_file.read()

                            os.makedirs(temp_dir, exist_ok=True)
                            with tempfile.NamedTemporaryFile(
                                mode="wb", delete=False, suffix=".bpmn", dir=temp_dir
                            ) as temp_file:
                                temp_file.write(contents)

                            bpmn_graph = read_bpmn(temp_file.name)
                            process_model = promoai.generate_model_from_bpmn(bpmn_graph)
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        elif file_extension == "pnml":
                            contents = uploaded_file.read()

                            os.makedirs(temp_dir, exist_ok=True)
                            with tempfile.NamedTemporaryFile(
                                mode="wb", delete=False, suffix=".pnml", dir=temp_dir
                            ) as temp_file:
                                temp_file.write(contents)
                            pn, im, fm = read_pnml(temp_file.name)
                            process_model = promoai.generate_model_from_petri_net(pn)

                            shutil.rmtree(temp_dir, ignore_errors=True)

                        else:
                            st.error(
                                body=f"Unsupported file format {file_extension}!",
                                icon="⚠️",
                            )
                            return

                        st.session_state["model_gen"] = process_model
                        st.session_state["feedback"] = []
                    except Exception as e:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        st.error(
                            body=f"Please upload a semi-block-structured model! The error message: {e}",
                            icon="⚠️",
                        )
                        return

    if "model_gen" in st.session_state and st.session_state["model_gen"]:

        st.success("Model generated successfully!", icon="🎉")

        col1, col2 = st.columns(2)

        try:
            with col1:
                with st.form(key="feedback_form"):
                    feedback = st.text_area("Feedback:", value="")
                    if st.form_submit_button(label="Update Model"):
                        try:
                            process_model = st.session_state["model_gen"]
                            process_model.update(
                                feedback,
                                api_key=api_key,
                                ai_model=ai_model_name,
                                ai_provider=provider,
                            )
                            st.session_state["model_gen"] = process_model
                        except Exception as e:
                            raise Exception("Update failed! " + str(e))
                        st.session_state["feedback"].append(feedback)

                    if len(st.session_state["feedback"]) > 0:
                        with st.expander("Feedback History", expanded=True):
                            i = 0
                            for f in st.session_state["feedback"]:
                                i = i + 1
                                st.write("[" + str(i) + "] " + f + "\n\n")

            with col2:
                st.write("Export Model")
                process_model_obj = st.session_state["model_gen"]
                powl = process_model_obj.get_powl()
                pn, im, fm = convert_to_petri_net(powl)
                bpmn = convert_to_bpmn(powl)
                download_1, download_2 = st.columns(2)
                with download_1:
                    bpmn_data = get_xml_string(
                        bpmn, parameters={"encoding": constants.DEFAULT_ENCODING}
                    )
                    st.download_button(
                        label="Download BPMN",
                        data=bpmn_data,
                        file_name="process_model.bpmn",
                        mime="application/xml",
                    )

                with download_2:
                    pn_data = export_petri_as_string(pn, im, fm)
                    st.download_button(
                        label="Download PNML",
                        data=pn_data,
                        file_name="process_model.pnml",
                        mime="application/xml",
                    )

            view_option = st.selectbox(
                "Select a view:", [v_type.value for v_type in ViewType]
            )

            image_format = str("svg").lower()
            if view_option == ViewType.POWL.value:
                from powl.visualization.powl import visualizer

                vis_str = visualizer.apply(powl)

            elif view_option == ViewType.PETRI.value:
                visualization = pn_visualizer.apply(
                    pn, im, fm, parameters={"format": image_format}
                )
                vis_str = visualization.pipe(format="svg").decode("utf-8")
            else:  # BPMN
                from pm4py.objects.bpmn.layout import layouter

                layouted_bpmn = layouter.apply(bpmn)
                visualization = bpmn_visualizer.apply(
                    layouted_bpmn, parameters={"format": image_format}
                )
                vis_str = visualization.pipe(format="svg").decode("utf-8")

            with st.expander("View Image", expanded=True):
                st.image(vis_str)

        except Exception as e:
            st.error(icon="⚠️", body=str(e))


def footer():
    style = """
        <style>
          .footer-container {
              position: fixed;
              left: 0;
              bottom: 0;
              width: 100%;
              text-align: center;
              padding: 15px 0;
              background-color: white;
              border-top: 2px solid lightgrey;
              z-index: 100;
          }

          .footer-text, .header-text {
              margin: 0;
              padding: 0;
          }
          .footer-links {
              margin: 0;
              padding: 0;
          }
          .footer-links a {
              margin: 0 10px;
              text-decoration: none;
              color: blue;
          }
          .footer-links img {
              vertical-align: middle;
          }
        </style>
        """

    foot = f"""
        <div class='footer-container'>
            <div class='footer-text'>
                Developed by
                <a href="https://www.linkedin.com/in/humam-kourani-98b342232/" target="_blank" style="text-decoration:none;">Humam Kourani</a>
                and
                <a href="https://www.linkedin.com/in/alessandro-berti-2a483766/" target="_blank" style="text-decoration:none;">Alessandro Berti</a>
                at the
                <a href="https://www.fit.fraunhofer.de/" target="_blank" style="text-decoration:none;">Fraunhofer Institute for Applied Information Technology FIT</a>.
            </div>
            <div class='footer-links'>
                <a href="https://doi.org/10.24963/ijcai.2024/1014" target="_blank">
                    <img src="https://img.shields.io/badge/ProMoAI:%20Process%20Modeling%20with%20Generative%20AI-gray?logo=googledocs&logoColor=white&labelColor=red" alt="ProMoAI Paper">
                </a>
                <a href="mailto:humam.kourani@fit.fraunhofer.de?cc=a.berti@pads.rwth-aachen.de;" target="_blank">
                    <img src="https://img.shields.io/badge/Email-gray?logo=minutemailer&logoColor=white&labelColor=green" alt="Email Humam Kourani">
                </a>
            </div>
        </div>
        """

    st.markdown(style, unsafe_allow_html=True)
    st.markdown(foot, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="ProMoAI", page_icon="🤖")
    footer()
    run_app()

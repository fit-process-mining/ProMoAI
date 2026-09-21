import subprocess
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def inject_css() -> None:
    with open("styles.css", "r") as file:
        css_file = file.read()
    st.markdown(
        f"""
    <style>
    {css_file}
    </style>
    """,
        unsafe_allow_html=True,
    )


def _launch_via_streamlit() -> None:
    script_path = Path(__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script_path)],
        check=False,
    )
    raise SystemExit(result.returncode)


def run_app():
    pg = st.navigation(
        [
            st.Page(
                "setup_page.py",
                title="Setup",
                icon=":material/settings_suggest:",
                default=True,
            ),
            st.Page("promoai_page.py", title="ProMoAI", icon=":material/account_tree:"),
            st.Page("pmax.py", title="PMAx", icon=":material/query_stats:"),
        ]
    )
    pg.run()


def sidebar_info():
    if "resettable" not in st.session_state:
        st.session_state["resettable"] = False

    with st.sidebar:
        if st.session_state["resettable"]:
            st.markdown(
                "<div class='sidebar-header'>Session</div>", unsafe_allow_html=True
            )

            if st.sidebar.button("🔄 Reset PMAx", use_container_width=True):
                st.session_state["setup_complete"] = False
                if "uploaded_log" in st.session_state:
                    del st.session_state["uploaded_log"]
                if "uploaded_log_path" in st.session_state:
                    del st.session_state["uploaded_log_path"]
                if "agent_state" in st.session_state:
                    del st.session_state["agent_state"]
                if "artifact_session_dir" in st.session_state:
                    del st.session_state["artifact_session_dir"]
                if "messages" in st.session_state:
                    del st.session_state["messages"]
                if "pdf_bytes" in st.session_state:
                    del st.session_state["pdf_bytes"]
                if "pdf_signature" in st.session_state:
                    del st.session_state["pdf_signature"]
                st.session_state["resettable"] = False
                st.rerun()

        # ---------- RESOURCES (TOP) ----------
        st.markdown(
            "<div class='sidebar-header'>Resources</div>", unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <a class="sidebar-link" href="https://doi.org/10.24963/ijcai.2024/1014" target="_blank">
                    <span style="font-size:1.2rem;">📄</span>
                    <div>
                        <span class="card-text-main">ProMoAI Paper</span>
                        <span class="card-text-sub">IJCAI 2024</span>
                    </div>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <a class="sidebar-link" href="https://doi.org/10.1007/978-3-032-28274-3_26" target="_blank">
                    <span style="font-size:1.2rem;">📄</span>
                    <div>
                        <span class="card-text-main">PMAx Paper</span>
                        <span class="card-text-sub">EMMSAD 2026</span>
                    </div>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """

            <div class='sidebar-header'>Support</div>

            <div class="sidebar-card">
                <a class="sidebar-link" href="mailto:humam.kourani@fit.fraunhofer.de;a.berti@pads.rwth-aachen.de;anton.antonov@fit.fraunhofer.de?subject=ProMoAI Inquiry">
                    <span style="font-size:1.2rem;">✉️</span>
                    <div>
                        <span class="card-text-main">Contact Developers</span>
                    </div>
                </a>
            </div>

            <div class='sidebar-header'>Developed at</div>

            <a href="https://www.fit.fraunhofer.de/" target="_blank" class="branding-badge">
                <img src="https://www.fit.fraunhofer.de/content/dam/fit/fit.svg">
                <div class="branding-text">
                    Fraunhofer Institute for Applied<br>Information Technology FIT
                </div>
            </a>

            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        _launch_via_streamlit()
    inject_css()
    st.set_page_config(page_title="ProMoAI", page_icon="🤖")

    sidebar_info()
    run_app()

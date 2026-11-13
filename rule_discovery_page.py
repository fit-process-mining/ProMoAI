"""Rule-based process discovery within the ProMoAI Streamlit interface."""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import promoai
import streamlit as st
from constants import MAX_FILE_SIZE
from powl import import_event_log

from promoai.general_utils.upload_utils import write_uploaded_file_to_staging
from promoai.prompting.declarative_rules import (
    parse_declare_rules,
    query_llm_for_declare_rules,
    refine_declare_rules,
)


REPAIR_OPTIONS = {
    "Event-level repair": "event_level",
    "Trace-level repair": "trace_level",
    "Edit-distance repair": "edit_distance",
    "No repair": "none",
}


def _rule_label(rule) -> str:
    name = "Init" if rule.name == "Initialization" else rule.name
    return f"{name}({', '.join(repr(label) for label in rule.args)})"


def _reset_rules() -> None:
    for key in (
        "rules_code",
        "rules_applied_code",
        "rules_normalized_code",
        "rules_candidates",
        "rules_selected",
        "rules_model_signature",
        "rules_enforced",
        "rules_feedback",
    ):
        st.session_state.pop(key, None)
    st.session_state["model_gen"] = None
    st.session_state["feedback"] = []


def _load_uploaded_log(uploaded_file) -> pd.DataFrame:
    temp_path = write_uploaded_file_to_staging(uploaded_file)
    try:
        return import_event_log(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _get_log(uploaded_file):
    if uploaded_file is None:
        if st.session_state.pop("rules_log_signature", None) is not None:
            st.session_state.pop("rules_event_log", None)
            _reset_rules()
        return None

    signature = (
        uploaded_file.name,
        hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
    )
    if signature != st.session_state.get("rules_log_signature"):
        try:
            log = _load_uploaded_log(uploaded_file)
        except Exception as exc:
            st.session_state.pop("rules_event_log", None)
            st.session_state.pop("rules_log_signature", None)
            _reset_rules()
            st.error(f"Could not import the event log: {exc}")
            return None
        if log.empty:
            st.session_state.pop("rules_event_log", None)
            st.session_state.pop("rules_log_signature", None)
            _reset_rules()
            st.error("The uploaded event log contains no events.")
            return None
        st.session_state["rules_event_log"] = log
        st.session_state["rules_log_signature"] = signature
        _reset_rules()
        st.session_state.pop("rules_column_signature", None)
        for key in (
            "rules_activity_column",
            "rules_case_column",
            "rules_timestamp_column",
        ):
            st.session_state.pop(key, None)
    return st.session_state.get("rules_event_log")


def _validate_log(log: pd.DataFrame, activity_key: str, case_id_key: str):
    if activity_key == case_id_key:
        raise ValueError("The activity and case ID columns must be different.")
    if log[activity_key].isna().any() or log[case_id_key].isna().any():
        raise ValueError(
            "The selected activity and case ID columns contain missing values."
        )
    labels = log[activity_key].unique().tolist()
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("All activity labels must be nonempty strings.")
    return sorted(labels)


def _validation_log(log: pd.DataFrame, completion_only: bool) -> pd.DataFrame:
    if completion_only and "lifecycle:transition" in log.columns:
        completed = log["lifecycle:transition"].isin(
            ("complete", "Complete", "COMPLETE")
        )
        if completed.any():
            return log.loc[completed]
    return log


def _apply_rules(
    code: str, activities: list[str], *, update_editor: bool = False
) -> None:
    normalized, rules = parse_declare_rules(code, activities)
    if update_editor:
        st.session_state["rules_code"] = normalized
    st.session_state["rules_applied_code"] = (
        normalized if update_editor else code.strip()
    )
    st.session_state["rules_normalized_code"] = normalized
    st.session_state["rules_candidates"] = rules
    st.session_state["rules_selected"] = [_rule_label(rule) for rule in rules]
    st.session_state["model_gen"] = None
    st.session_state["feedback"] = []
    st.session_state.pop("rules_model_signature", None)


def render_rule_based_discovery():
    st.markdown("### Rule-based POWL Discovery")
    st.caption(
        "Use natural-language DECLARE constraints or enter rules directly, "
        "then discover a POWL model that satisfies the selected rules."
    )

    uploaded_file = st.file_uploader(
        "Event log",
        type=["xes", "gz", "csv"],
        key="rules_uploaded_log",
        max_upload_size=MAX_FILE_SIZE,
    )
    log = _get_log(uploaded_file)
    if log is None:
        st.info("Upload an event log to start rule-based discovery.")
        return

    columns = log.columns.tolist()
    default_activity = (
        columns.index("concept:name") if "concept:name" in columns else 0
    )
    default_case = (
        columns.index("case:concept:name") if "case:concept:name" in columns else 0
    )
    cols = st.columns(2)
    with cols[0]:
        activity_key = st.selectbox(
            "Activity column",
            options=columns,
            index=default_activity,
            key="rules_activity_column",
        )
    with cols[1]:
        case_id_key = st.selectbox(
            "Case ID column",
            options=columns,
            index=default_case,
            key="rules_case_column",
        )
    timestamp_choices = ["(Use file order)"] + columns
    timestamp_key = st.selectbox(
        "Timestamp column",
        options=timestamp_choices,
        index=(
            timestamp_choices.index("time:timestamp")
            if "time:timestamp" in columns
            else 0
        ),
        key="rules_timestamp_column",
    )
    timestamp_key = None if timestamp_key == "(Use file order)" else timestamp_key
    try:
        activities = _validate_log(log, activity_key, case_id_key)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"{log[case_id_key].nunique()} cases · {len(log):,} events · "
        f"{len(activities)} activities"
    )

    activity_signature = (activity_key, case_id_key, timestamp_key)
    if activity_signature != st.session_state.get("rules_column_signature"):
        st.session_state["rules_column_signature"] = activity_signature
        _reset_rules()

    st.markdown("#### Extract or enter DECLARE rules")
    description = st.text_area(
        "Process description",
        key="rules_description",
        height=140,
        placeholder="Describe which activities are required, ordered, or mutually exclusive...",
    )
    credentials = st.session_state.get("llm_credentials")
    if credentials is None:
        st.caption(
            "Configure an AI provider on the Setup page to extract rules "
            "from text. You can also enter rules manually below."
        )

    if st.button("Extract rules from description", disabled=credentials is None):
        if not description.strip():
            st.warning("Enter a process description first.")
        else:
            try:
                with st.spinner("Extracting DECLARE rules..."):
                    _, corrections, code = query_llm_for_declare_rules(
                        description, activities, credentials
                    )
                    _apply_rules(code, activities, update_editor=True)
                if corrections:
                    st.toast(f"Resolved {corrections} rule-format correction(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Rule extraction failed: {exc}")

    if st.session_state.get("rules_candidates"):
        feedback = st.text_area(
            "Refine extracted rules (optional)",
            key="rules_feedback",
            placeholder="e.g., Payment must immediately follow approval.",
        )
        if st.button(
            "Refine rules with AI",
            disabled=(
                credentials is None or not feedback.strip() or not description.strip()
            ),
        ):
            try:
                with st.spinner("Refining DECLARE rules..."):
                    _, _, revised_code = refine_declare_rules(
                        description,
                        activities,
                        st.session_state["rules_normalized_code"],
                        feedback,
                        credentials,
                    )
                    _apply_rules(revised_code, activities, update_editor=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Rule refinement failed: {exc}")

    st.session_state.setdefault("rules_code", "")
    code = st.text_area(
        "DECLARE rules (editable)",
        key="rules_code",
        height=170,
        placeholder="rule1 = Init('Register')\nrule2 = Response('Register', 'Complete')",
        help="Use one supported DECLARE rule call per line. Activity names must match the uploaded log.",
    )
    if st.button("Apply rules"):
        try:
            _apply_rules(code, activities)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if not st.session_state.get("rules_candidates"):
        st.info("Extract rules or paste DECLARE rules and click Apply rules.")
        return
    if code.strip() != st.session_state.get("rules_applied_code", ""):
        st.warning(
            "The rules have changed. Click Apply rules before discovering a model."
        )
        return

    candidates = st.session_state["rules_candidates"]
    completion_only = st.checkbox(
        "Keep only completion events",
        value=True,
        help="When lifecycle transitions are available, use completed events for validation and discovery.",
        key="rules_completion_only",
    )
    current_log = _validation_log(log, completion_only)
    try:
        results = promoai.validate_event_log_rules(
            current_log,
            candidates,
            activity_key=activity_key,
            case_id_key=case_id_key,
            timestamp_key=timestamp_key,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Could not validate rules: {exc}")
        return

    st.markdown("#### Rule validation")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rule": _rule_label(result.rule),
                    "Support": round(result.support, 3),
                    "Confidence": round(result.confidence, 3),
                    "Activated traces": result.activated_traces,
                    "Violations": result.violated_traces,
                }
                for result in results
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Support includes vacuous satisfaction. Check activated traces "
        "when interpreting confidence."
    )
    rule_map = {_rule_label(rule): rule for rule in candidates}
    selected = st.multiselect(
        "Rules to enforce",
        options=list(rule_map),
        key="rules_selected",
    )
    st.download_button(
        "Download rules",
        data=st.session_state["rules_normalized_code"] + "\n",
        file_name="declare_rules.py",
        mime="text/plain",
    )

    st.markdown("#### Discovery settings")
    left, right = st.columns(2)
    with left:
        repair_name = st.selectbox(
            "Repair strategy", options=list(REPAIR_OPTIONS), key="rules_repair_mode"
        )
    with right:
        noise_threshold = st.slider(
            "Noise threshold",
            min_value=0.0,
            max_value=0.95,
            value=0.0,
            step=0.05,
            key="rules_noise_threshold",
        )

    signature = (
        st.session_state.get("rules_log_signature"),
        activity_signature,
        st.session_state["rules_applied_code"],
        tuple(selected),
        repair_name,
        noise_threshold,
        completion_only,
    )
    if st.session_state.get("rules_model_signature") not in (None, signature):
        st.session_state["model_gen"] = None
        st.session_state["feedback"] = []
        st.session_state.pop("rules_model_signature", None)
        st.session_state.pop("rules_enforced", None)

    if st.button("Discover rule-constrained model", type="primary"):
        if not selected:
            st.warning("Select at least one rule to enforce.")
            return
        try:
            with st.spinner("Discovering rule-constrained POWL model..."):
                model = promoai.generate_model_from_event_log_with_rules(
                    log,
                    [rule_map[label] for label in selected],
                    repair_mode=REPAIR_OPTIONS[repair_name],
                    noise_threshold=noise_threshold,
                    verify_rules=True,
                    activity_key=activity_key,
                    case_id_key=case_id_key,
                    timestamp_key=timestamp_key,
                    keep_only_completion_events=completion_only,
                )
            st.session_state["model_gen"] = model
            st.session_state["feedback"] = []
            st.session_state["rules_model_signature"] = signature
            st.session_state["rules_enforced"] = [rule_map[label] for label in selected]
            st.rerun()
        except Exception as exc:
            st.error(f"Rule-constrained discovery failed: {exc}")

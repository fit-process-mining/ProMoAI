from typing import Any, Dict, List, Tuple, Union

from pm4py.objects.log.obj import EventLog

from pm4py.objects.petri_net.obj import Marking, PetriNet

from promoai.general_utils.artifact_store import (
    append_manifest_entry,
    create_analysis_session,
)
import pandas as pd
import pm4py
from promoai.agents.utils import transform_dataframe_for_llms



class ProcessState(dict):
    user_request: List[str]

    # ---- Data Objects ---- #

    initial_event_log: EventLog

    event_log: EventLog

    previous_code: str
    
    initial_process_model: Union[Tuple[Any, Any, Any], None]

    process_model : Union[Tuple[PetriNet, Marking, Marking], None]
    # --- Meta Data and Abstractions --- #
    log_abstraction: str

    saved_artifacts: Dict[str, Any]

    context: List[str]

    log_actions: Dict[str, List[str]]

    final_report: List[Dict[str, Any]]

    messages_ana: List[Any]

    messages_eng: List[Any]

    extracted_statistics: Dict[str, Any]

    analysis_data: Dict[str, Any]

    # To keep track of sent artifacts to the analyst so we spare tokens
    sent_artifacts: List[str]

    artifact_session_dir: str
    source_log_path: str | None

    def __init__(
        self,
        user_request: str,
        event_log: Union[EventLog, None],
        process_model : Union[Tuple[Any, Any, Any], None],
        artifact_session_dir: str | None = None,
        source_log_path: str | None = None,
    ):
        super().__init__()
        self["user_request"] = [user_request]
        if event_log is None:
            raise ValueError(
                "Event log cannot be None. Please provide a valid event log to initialize the ProcessState."
            )
        self["event_log"] = event_log
        self["initial_event_log"] = event_log
        self["artifact_session_dir"] = artifact_session_dir or create_analysis_session(
            "pmax"
        )
        self["process_model"] = process_model
        self["initial_process_model"] = process_model
        self["source_log_path"] = source_log_path
        self["log_abstraction"] = self.generate_log_abstraction()
        self["model_abstraction"] = self.generate_model_abstraction()
        self["saved_artifacts"] = {}
        self["context"] = []
        self["previous_code"] = ""
        self["log_actions"] = {}
        self["final_report"] = []
        self["messages_ana"] = []
        self["messages_eng"] = []
        self["extracted_statistics"] = {}
        self["analysis_data"] = {}
        self["sent_artifacts"] = []
        append_manifest_entry(
            self["artifact_session_dir"],
            category="session",
            file_path=self["artifact_session_dir"],
            description="Initialized PMAx analysis session.",
            artifact_type="session",
            extra={"source_log_path": source_log_path},
        )

    def __str__(self):
        return (
            f"ProcessState(\n"
            f"  user_request={self['user_request']},\n"
            f"  log_abstraction={self['log_abstraction']},\n"
            f"  extracted_statistics={self['extracted_statistics']},\n"
            f"  process_model={'Available' if self['process_model'] else 'N/A'},\n"
            f"  analysis_data={self['analysis_data']},\n"
            f"  log_actions={self['log_actions']},\n"
            f"  final_report={self['final_report'][:100]}..., \n"
            f"  messages_ana=[{len(self['messages_ana'])} messages]\n"
            f"  messages_eng=[{len(self['messages_eng'])} messages]\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()
    

    def generate_process_abstraction(self):
        if self["process_model"] is None:
            return "Missing process model."
        msg = f"Process Model Abstraction:\n"
        msg += pm4py.llm.abstract_petri_net(self["process_model"])
        return msg
        
    def generate_log_abstraction(self):
        df = self["event_log"]

        if df is None or len(df) == 0:
            return "Missing event log."

        # Basic Stats
        msg = f"Event Log Abstraction:\n"
        msg += f"- Total Events: {len(df)}\n"
        msg += f"- Total Cases: {df['case:concept:name'].nunique() if 'case:concept:name' in df.columns else 'N/A'}\n"
        if "concept:name" in df.columns:
            msg += f"- Unique activities: {df['concept:name'].dropna().unique().tolist()}\n"
        msg += "- Columns & Samples:\n"

        for col in df.columns:
            col_type = df[col].dtype

            sample_data = df[col].dropna().unique()[:10].tolist()

            if "datetime" in str(col_type):
                sample_data = [str(x) for x in sample_data]

            msg += f"  * '{col}' (Type: {col_type}): {sample_data}\n"

        return msg
    
    def generate_model_abstraction(self):
        if self["process_model"] is None:
            return None
        msg = f"The initial process model looks like this:\n"
        msg += pm4py.llm.abstract_petri_net(*self["process_model"])
        return msg
    
    def inject_context_table(self, table : pd.DataFrame):
        parsed_table = transform_dataframe_for_llms(table)
        self.add_context(parsed_table)

    def add_context(self, description: str):
        self["context"].append(description)

    def log_action(self, description: str):
        key = "Preprocessor Node"
        if key not in self["log_actions"]:
            self["log_actions"][key] = []
        self["log_actions"][key].append(description)

    def save_model(self, model: Tuple[Any, Any, Any]):
        self["process_model"] = model

    def update_artifacts(self, pathway: str, artifact_description: Any, data: Any):
        self["saved_artifacts"][pathway] = (artifact_description, data)

    def wipe_artifacts(self):
        self["saved_artifacts"] = {}

    def flush_context(self):
        self["context"] = []

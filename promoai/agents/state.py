from typing import Any, Dict, List, Tuple

from pm4py.objects.log.obj import EventLog


class ProcessState(dict):
    user_request: List[str]

    # ---- Data Objects ---- #
    event_log: EventLog

    previous_code: str
    # --- Meta Data and Abstractions --- #
    log_abstraction: str

    discovered_model: Tuple[Any, Any, Any]

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

    def __init__(self, user_request: str, event_log: EventLog):
        super().__init__()
        self["user_request"] = [user_request]
        if event_log is None:
            raise ValueError(
                "Event log cannot be None. Please provide a valid event log to initialize the ProcessState."
            )
        self["event_log"] = event_log
        self["log_abstraction"] = self.generate_log_abstraction()
        self["discovered_model"] = None
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

    def __str__(self):
        return (
            f"ProcessState(\n"
            f"  user_request={self['user_request']},\n"
            f"  log_abstraction={self['log_abstraction']},\n"
            f"  extracted_statistics={self['extracted_statistics']},\n"
            f"  discovered_model={'Available' if self['discovered_model'] else 'None'},\n"
            f"  analysis_data={self['analysis_data']},\n"
            f"  log_actions={self['log_actions']},\n"
            f"  final_report={self['final_report'][:100]}..., \n"
            f"  messages_ana=[{len(self['messages_ana'])} messages]\n"
            f"  messages_eng=[{len(self['messages_eng'])} messages]\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()

    def generate_log_abstraction(self):
        return f"Event log with {len(self['event_log'])} events and the following attributes: {list(self['event_log'].columns) if len(self['event_log']) > 0 else 'No events'}"

    def add_context(self, description: str):
        self["context"].append(description)

    def log_action(self, description: str):
        key = "Preprocessor Node"
        if key not in self["log_actions"]:
            self["log_actions"][key] = []
        self["log_actions"][key].append(description)

    def save_model(self, model: Tuple[Any, Any, Any]):
        self["discovered_model"] = model

    def update_artifacts(self, pathway: str, artifact_description: Any, data: Any):
        self["saved_artifacts"][pathway] = (artifact_description, data)

    def wipe_artifacts(self):
        self["saved_artifacts"] = {}

    def flush_context(self):
        self["context"] = []

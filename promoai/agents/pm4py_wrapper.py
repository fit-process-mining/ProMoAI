import os
import re
from typing import Tuple, Union, Any

import pandas as pd
import pm4py
import powl
from pm4py.objects import petri_net as PNet
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from powl import convert_to_petri_net
import promoai.agents.utils as utils
from promoai.agents.state import ProcessState
from promoai.agents.utils import transform_dataframe_for_llms
from promoai.general_utils.artifact_store import (
    append_manifest_entry,
    create_managed_path,
)
from promoai.model_generation.code_extraction import execute_code_and_get_variable
from promoai.general_utils.llm_connection import (
    LLMConnection
)
from promoai.model_generation.llm_model_generator import LLMProcessModelGenerator
from powl.conversion.to_powl.from_pn.converter import convert_workflow_net_to_powl 

class LLMClient:
    def __init__(self, credentials : LLMConnection):
        self.credentials = credentials

    def from_description(self, process_description):
        api_key = self.credentials.api_key
        llm = self.credentials.llm_name
        provider = self.credentials.ai_provider
        args = self.credentials.args
        return LLMProcessModelGenerator.from_description(process_description, api_key, llm, provider, False, args)
    
    def from_pnet(self, pnet : Tuple[Any, Any, Any], description : str):
        powl = convert_workflow_net_to_powl(pnet[0])
        api_key = self.credentials.api_key
        llm = self.credentials.llm_name
        provider = self.credentials.ai_provider
        args = self.credentials.args
        _, pnet = LLMProcessModelGenerator(None, [], None).edit_code(feedback=description, powl_model = powl, api_key=api_key, ai_model=llm, ai_provider=provider, llm_args=args)

        return pnet

class PM4PYWrapper:
    def __init__(self, state: ProcessState, client : Union[LLMClient, None]):
        # ==== Load event log ==== #
        self.event_log = state["event_log"]
        self._initial_event_log = state["initial_event_log"]
        self.client = client
        # used to detect potential leaks
        self._raw_column_fingerprints = {
            hash(tuple(sorted(self.event_log[col].astype(str).unique()))): col
            for col in self.event_log.columns
        }
        # ========================= #
        self.pnet = state.get("process_model", None)
        self._initial_process_model = state.get("initial_process_model", None)
        self.state = state
    @property
    def initial_process_model(self):
        return self._initial_process_model
    @property
    def initial_event_log(self):
        return self._initial_event_log
    @staticmethod
    def get_API_summary() -> str:
        return """
        You have access to a variable `api` which is an instance of the Process Mining Preprocessing Engine.

        AVAILABLE METHODS:
        1. Filtering (Updates the current log in-place): \n
           - api.filter_time_range(start_date: str, end_date: str) -> "YYYY-MM-DD" \n
           - api.filter_attribute(column: str, value: str) \n
           - api.filter_pandas_query(query: str) -> e.g. "amount > 500" \n

        2. Abstraction (Adds text summaries to the context which is passed down to the analyst node): \n
           - api.get_dfg_summary() -> Returns a summary (STRING) of the Directly-Follows Graph (Markovian abstraction of the event log). \n
           - api.get_model_summary() -> Returns a summary (STRING) of the discovered process model (if discovered, otherwise, will be discovered first) (e.g., Petri net).\n
           - api.get_variant_summary() -> Returns a summary (STRING) of the most common variants (unique sequences of activities) in the log. \n
           - api.get_case_summary() -> Returns a summary (STRING) of individual cases, including common patterns and outliers. \n

        3. Mining & Analysis:\n
           - api.discover_process_model() -> works on ``api.event_log``, returns nothing, updates internal state, i.e., ``api.process_model`` with a discovered Petri net model based on the event log and saves visualization of it. \n
           - api.cc_alignments(event_log, process_model) -> returns conformance checking results based on alignments, i.e., a tuple of fitness, precision, F1 using **the provided process model and event log**. \n
           - api.cc_token_based_replay(event_log, process_model) -> returns conformance checking results based on token-based replay, i.e., a tuple of fitness, precision, F1 **using the provided event log and process model**. \n
           - api.discover_from_text(description : str) -> saves a Petri net (process model) as ``api.process_model`` in state of a process described with text. \n
           - api.edit_model(description : str) -> modifies the discovered process model (from text or data) solely using textual description and updates the saved process model. \n
        \n
        4. Visualization:
            - api.save_pnet() -> Saves the discovered Petri net model (from ``api.process_model``) as visualization for further analysis. \n
            - api.save_dfg() -> Saves the Directly-Follows Graph (using the current event log ``api.event_log``) as visualization for further analysis. \n
            - api.save_visualization(fig, description, data) -> Saves a given visualization figure with a description for context Additionally, add the data used for its generation. \n
                Use this to save all kinds of visualizations, including generated via matplotlib, seaborn or plotly. \n
                Additionally, make sure that you ALWAYS provide description to give context to the saved visualization, this will be used in the final report, as well as the data used for its generation in form of a dataframe/dictionary. \n
            - api.save_dataframe(df, description) -> Saves a given dataframe to a specified file path with a description for context. \n
                Use this to pass down any kind of dataframes, including intermediate data manipulations or results of process mining algorithms. \n
                There is NO need to pass down the final event log with this method, use the return variable `final_event_log` for that. \n
                Note that saving whole columns of the event log or the entire event log will trigger an error, **focus on the requested analysis** when saving dataframes.
        \n
        RULES: \n
        - You can use only the provided methods, as well as matplotlib, seaborn, plotly, numpy and pandas for any additional data manipulation or visualization needs. \n
        - Ensure that any code you generate adheres to the whitelisted libraries and can compile without errors. \n
        - If the user asks to filter (e.g. "only 2023", "remove attribute X"), write Python code to call `api.filter...`. or pandas \n
        - If you make use of other methods from allowed libraries apart from the provided api, make sure to include the necessary import statements in the generated code. \n
        - Always use the save_visualization method to save any visualization, built-in methods in matplotlib or seaborn are disabled. \n
        - Always use the save_dataframe method to save any dataframe, AVOID built-in methods in pandas. \n
        - Whenever asked to edit a process model, use the `edit_model` method which takes a textual description of the required edit, and modifies the current process model solely based on the provided textual description. \n
        - If `edit_model` fails, you can use the standard `discover_from_text` method but provide a detailed textual description of the original model (use abstraction method) and the required edit. \n
        """

    def _add_context(self, description: str):
        self.state.add_context(description)

    def _log_action(self, description: str):
        # Check if it already exists, if this is the case, then append it to a list
        self.state.log_action(description)

    def save_dataframe(self, df: pd.DataFrame, description: str):
        if description is None or description.strip() == "":
            raise ValueError(
                "Description for the saved dataframe cannot be empty. Please provide a meaningful description to give context to the saved dataframe."
            )
        # to make sure that data isn't leaked
        potential_column_leak = len(df) == len(self.event_log)
        if potential_column_leak:
            for col in df.columns:
                artifact_col_sig = hash(tuple(sorted(df[col].astype(str).unique())))
                if artifact_col_sig in self._raw_column_fingerprints:
                    raise Exception(
                        f"Artifact {description} contains raw event log content in {col}. \
                                    Avoid saving raw data as artifacts, instead, focus on the analysis."
                    )

        data_preview = transform_dataframe_for_llms(df)
        file_path = create_managed_path(
            self.state["artifact_session_dir"],
            "dataframes",
            description,
            ".csv",
            prefix="dataframe",
        )
        self.state.update_artifacts(file_path, description, data_preview)
        self._add_context(f"Dataframe saved: {description}")
        df.to_csv(file_path, index=False)
        append_manifest_entry(
            self.state["artifact_session_dir"],
            category="dataframes",
            file_path=file_path,
            description=description,
            artifact_type="dataframe",
            data_preview=data_preview,
            extra={"rows": len(df), "columns": list(df.columns)},
        )

    def save_visualization(self, fig, description: str, data):
        if description is None or description.strip() == "":
            raise ValueError(
                "Description for the visualization cannot be empty. Please provide a meaningful description to give context to the saved visualization."
            )
        data = (
            transform_dataframe_for_llms(data)
            if isinstance(data, pd.DataFrame)
            else data
        )
        file_path = create_managed_path(
            self.state["artifact_session_dir"],
            "visualizations",
            description,
            ".png",
            prefix="visual",
        )
        self.state.update_artifacts(file_path, description, data)
        self._add_context(f"Visualization generated: {description}")
        # Export the visualization to the specified file path
        if hasattr(fig, "savefig"):
            fig.savefig(file_path, bbox_inches="tight")
        elif hasattr(fig, "render"):
            base_path = os.path.splitext(file_path)[0]
            fmt = os.path.splitext(file_path)[1].replace(".", "") or "png"
            fig.render(base_path, format=fmt, cleanup=True)
        elif hasattr(fig, "write_image"):
            fig.write_image(file_path)
        elif str(type(fig)).find("plotly.graph_objs") != -1:
            fig.write_image(file_path)
        append_manifest_entry(
            self.state["artifact_session_dir"],
            category="visualizations",
            file_path=file_path,
            description=description,
            artifact_type="visualization",
            data_preview=data,
        )

    def save_dfg(self):
        dfg, start_activities, end_activities = pm4py.discover_dfg(self.event_log)

        fig = pm4py.visualization.dfg.visualizer.apply(dfg, log=self.event_log)
        self.save_visualization(
            fig,
            "Directly-Follows Graph (DFG)",
            data={
                "dfg": dfg,
                "start_activities": start_activities,
                "end_activities": end_activities,
            },
        )

    def save_pnet(self):
        if self.state["process_model"] is not None:
            # export it using pm4py's built-in visualizer
            net, im, fm = self.state["process_model"]
            gviz = pn_visualizer.apply(net, im, fm)
            self.save_visualization(
                gviz, "Discovered Petri Net", data=self.get_model_summary()
            )
        elif self.state["event_log"] is not None:
            # Discover it
            self.discover_process_model()
            net, im, fm = self.state["discovered_model"]
            gviz = pn_visualizer.apply(net, im, fm)
            self.save_visualization(
                gviz, "Discovered Petri Net", data=self.get_model_summary()
            )
        else:
            raise ValueError(
                "No event log avaiable in the state to discover a process model. Cannot generate Petri net visualization."
            )

    # ===== Preprocessing Methods ===== #
    # --- STEP 1: PREPROCESSOR (Filtering) ---
    def filter_time_range(self, start_date: str, end_date: str):
        """Filters log by timeframe (YYYY-MM-DD). Updates self.df inplace."""
        self.event_log = pm4py.filter_time_range(
            self.event_log, start_date, end_date, mode="traces_contained"
        )
        msg = f"Filtered log from {start_date} to {end_date}. Remaining cases: {len(self.event_log)}"
        return self._log_action(msg)

    def filter_attribute(self, column: str, value: str):
        """Filters specific attribute (e.g., org:resource == 'UserA')."""
        self.event_log = pm4py.filter_event_attribute_values(
            self.event_log, column, [value], level="case"
        )
        msg = f"Filtered cases where {column} contains {value}."
        return self._log_action(msg)

    def filter_pandas_query(self, query: str):
        """Allows complex filtering via pandas query string."""
        if not isinstance(self.event_log, pd.DataFrame):
            self.event_log = pm4py.convert_to_dataframe(self.event_log)
        self.event_log = self.event_log.query(query)
        msg = f"Applied pandas query filter: {query}"
        return self._log_action(msg)

    # ===== LLM-based Abstractions ===== #

    def get_dfg_summary(self):
        summary = pm4py.llm.abstract_dfg(self.event_log)
        return summary

    def get_model_summary(self):
        if self.pnet is None:
            # Discover the process model if not already done
            self.discover_process_model()
        net, im, fm = self.pnet
        model_summary = pm4py.llm.abstract_petri_net(net, im, fm)
        return model_summary

    def get_variant_summary(self):
        variant_summary = pm4py.llm.abstract_variants(self.event_log)
        return variant_summary

    def get_case_summary(self):
        try:
            case_summary = pm4py.llm.abstract_case(self.event_log)
        except Exception as e:
            raise Exception(f"The log should be preprocessed so that the standard attributes, i.e., case:concept:name (case id), concept:name (activity) and time:timestamp (timestamp) are present. Error details: {str(e)}")
        return case_summary

    # ===== Process Mining Algorithms ===== #

    def discover_process_model(self):
        if self.event_log is None or len(self.event_log) == 0:
            raise Exception("No event log available")
        self.powl = powl.discover(self.event_log)
        net, im, fm = convert_to_petri_net(self.powl)
        self.pnet = (net, im, fm)
        self.state.save_model((net, im, fm))
    
    def discover_from_text(self, description : str):
        pnet = self.client.from_description(description).get_petri_net()
        self.state.save_model(pnet)
        self._log_action(f"Discovered process model from {description}")

    def edit_model(self, description : str):
        """
        Modifies the current process model based on a textual description
        """
        if self.pnet is None:
            try:
                self.discover_process_model()
            except Exception:
                raise Exception("No process model is discovered, missing event log, discover from text first.")
        self.state.save_model(self.client.from_pnet(self.pnet, description))
        self._log_action(f"Edited process model using {description}")
        
    def cc_alignments(self, event_log, net, im, fm) -> Tuple[float, float, float]:
        fitness = pm4py.conformance_diagnostics_alignments(event_log, net, im, fm)
        precision = pm4py.precision_alignments(event_log, net, im, fm)
        f1 = (
            2 * (fitness * precision) / (fitness + precision)
            if (fitness + precision) > 0
            else 0
        )

        self._add_context(
            f"Conformance Checking - Alignments: Fitness: {fitness}, Precision: {precision}, F1 Score: {f1}"
        )
        return (fitness, precision, f1)

    def cc_token_based_replay(self, net: PNet, im, fm) -> Tuple[float, float, float]:
        fitness = pm4py.conformance_diagnostics_token_based_replay(
            self.event_log, net, im, fm
        )
        precision = pm4py.precision_token_based_replay(self.event_log, net, im, fm)
        f1 = (
            2 * (fitness * precision) / (fitness + precision)
            if (fitness + precision) > 0
            else 0
        )
        self._add_context(
            f"Conformance Checking - Token-Based Replay: Fitness: {fitness}, Precision: {precision}, F1 Score: {f1}"
        )
        return (fitness, precision, f1)

    def code_extraction(self, code_snippet: str, args=None):
        """
        Extracts code from a given code snippet, removing any markdown formatting.
        """
        # Check that the code is wrapped in ```python ... ```
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, code_snippet, re.DOTALL)

        if not match:
            raise ValueError(
                "Code snippet is not properly formatted with ```python ... ```"
            )

        code = match.group(1).strip()
        utils._check_whitelisted_libraries(code)
        utils._check_code_for_compilation(code)
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pm4py

        namespace = {
            "api": self,
            "pd": pd,
            "np": np,
            "pm4py": pm4py,
            "plt": plt,
            "final_event_log": self.event_log,
        }

        return code, execute_code_and_get_variable(
            code, "final_event_log", namespace=namespace
        )

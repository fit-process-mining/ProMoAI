import datetime
import os
import re

import pandas as pd
import pm4py
import powl
from pm4py.objects import petri_net as PNet
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from powl import convert_to_petri_net

import promoai.agents.utils as utils
from promoai.agents.state import ProcessState
from promoai.agents.utils import transform_dataframe_for_llms
from promoai.model_generation.code_extraction import execute_code_and_get_variable


class PM4PYWrapper:
    def __init__(self, state: ProcessState):
        # ==== Load event log ==== #
        self.event_log = state["event_log"]
        # ========================= #
        self.process_model = state.get("discovered_model", None)
        self.state = state

    @staticmethod
    def get_API_summary() -> str:
        return """
        You have access to a variable `api` which is an instance of the Process Mining Preprocessing Engine.

        AVAILABLE METHODS:
        1. Filtering (Updates the log in-place):
           - api.filter_time_range(start_date: str, end_date: str) -> "YYYY-MM-DD"
           - api.filter_attribute(column: str, value: str)
           - api.filter_pandas_query(query: str) -> e.g. "amount > 500"

        2. Abstraction (Adds text summaries to the context which is passed down to the analyst node):
           - api.get_dfg_summary() -> Summary of the Directly-Follows Graph (Markovian abstraction of the event log).
           - api.get_model_summary() -> Summary of the discovered process model (if discovered, otherwise, will be discovered first) (e.g., Petri net).
           - api.get_variant_summary() -> Summary of the most common variants (unique sequences of activities) in the log.
           - api.get_case_summary() -> Summary of individual cases, including common patterns and outliers.

        3. Mining & Analysis:
           - api.discover_process_model() -> returns nothing, updates internal state with a discovered Petri net model based on the event log.
           - api.cc_alignments() -> returns conformance checking results based on alignments, i.e., a tuple of fitness, precision, F1.
           - api.cc_token_based_replay() -> returns conformance checking results based on token-based replay, i.e., a tuple of fitness, precision, F1.

        4. Visualization:
            - api.save_pnet() -> Saves the discovered Petri net model if available in the state, returns nothing. \n
            - api.save_dfg() -> Saves the Directly-Follows Graph (DFG) based on the event log in the state, returns nothing. \n
            - api.save_visualization(fig, description, data) -> Saves a given visualization figure with a description for context Additionally, add the data used for its generation.
                Use this to save all kinds of visualizations, including generated via matplotlib, seaborn or plotly.
                Additionally, make sure that you ALWAYS provide description to give context to the saved visualization, this will be used in the final report, as well as the data used for its generation in form of a dataframe/dictionary. \n
            - api.save_dataframe(df, description) -> Saves a given dataframe to a specified file path with a description for context.
                Use this to pass down any kind of dataframes, including intermediate data manipulations or results of process mining algorithms.
                There is NO need to pass down the final event log with this method, use the return variable `final_event_log` for that. \n

        RULES:
        - You can use only the provided methods, as well as matplotlib, seaborn, plotly, numpy and pandas for any additional data manipulation or visualization needs.
        - Ensure that any code you generate adheres to the whitelisted libraries and can compile without errors.
        - If the user asks to filter (e.g. "only 2023", "remove attribute X"), write Python code to call `api.filter...`. or pandas \n
        - If you make use of other methods from allowed libraries apart from the provided api, make sure to include the necessary import statements in the generated code. \n
        - Always use the save_visualization method to save any visualization, built-in methods in matplotlib or seaborn are disabled. \n
        - DO generate visualizations sparsely, as they cause significant overhead.
        - Always use the save_dataframe method to save any dataframe, AVOID built-in methods in pandas. \n

        """

    def _add_context(self, description: str):
        self.state.add_context(description)

    def _log_action(self, description: str):
        # Check if it already exists, if this is the case, then append it to a list
        self.state.log_action(description)

    def __preprocess_pathway(self, file_name: str, type) -> str:
        file_name = file_name.replace("..", "").replace("/", "").replace("\\", "")
        format = file_name.split(".")[-1]

        if format not in ["png", "pdf", "svg"] and type == "visualization":
            raise ValueError(
                "Unsupported file format. Please use one of the following: png, pdf, svg for images!"
            )
        if format not in ["csv", "parquet", "xes"] and type == "dataframe":
            raise ValueError(
                "Unsupported file format. Please use only csv for exporting dataframes!"
            )
        return file_name

    def save_dataframe(self, df: pd.DataFrame, description: str):
        if description is None or description.strip() == "":
            raise ValueError(
                "Description for the saved dataframe cannot be empty. Please provide a meaningful description to give context to the saved dataframe."
            )
        id_ = len(self.state["saved_artifacts"]) + 1
        file_name = f"dataframe_artifact_{id_}.csv"
        root_dir = os.getcwd()
        save_dir = os.path.join(root_dir, "temp", "dataframes")
        # add timestamp to save_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        # add timestamp to filename
        file_path = os.path.join(save_dir, file_name)
        # Check if the filepath exists, if it does, raise an error
        if os.path.exists(file_path):
            raise FileExistsError(
                f"The file {file_path} already exists. Please choose a different name to avoid overwriting."
            )
        self.state.update_artifacts(
            file_path, description, transform_dataframe_for_llms(df)
        )
        self._add_context(f"Dataframe saved: {description}")
        df.to_csv(file_path, index=False)

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
        id_ = len(self.state["saved_artifacts"]) + 1
        file_name = f"visual_artifact_{id_}.png"
        file_name = self.__preprocess_pathway(file_name, "visualization")
        root_dir = os.getcwd()
        save_dir = os.path.join(root_dir, "temp", "visualizations")
        # add timestamp to save_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            raise FileExistsError(
                f"The file {file_path} already exists. Please choose a different name to avoid overwriting."
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

    def save_pnet(self):
        if self.state["discovered_model"] is not None:
            # export it using pm4py's built-in visualizer
            net, im, fm = self.state["discovered_model"]
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

    def save_dfg(self):
        if self.state["event_log"] is not None:
            pm4py.discover_dfg(self.state["event_log"])
            self._add_context(f"DFG Visualization: {self.get_dfg_summary()}")
        else:
            raise ValueError(
                "Event log is not available in the state. Cannot generate DFG visualization."
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

    def get_dfg_summary(self) -> str:
        summary = pm4py.llm.abstract_dfg(self.event_log)
        return self._add_context(f"DFG Summary: {summary}")

    def get_model_summary(self) -> str:
        if self.pnet is None:
            # Discover the process model if not already done
            self.discover_process_model()
        net, im, fm = self.pnet
        return self._add_context(
            f"Model Summary: {pm4py.llm.abstract_petri_net(net, im, fm)}"
        )

    def get_variant_summary(self) -> str:
        return self._add_context(
            f"Variant Summary: {pm4py.llm.abstract_variants(self.event_log)}"
        )

    def get_case_summary(self) -> str:
        return self._add_context(
            f"Case Summary: {pm4py.llm.abstract_case(self.event_log)}"
        )

    # ===== Process Mining Algorithms ===== #

    def discover_process_model(self):
        self.powl = powl.discover(self.event_log)
        net, im, fm = convert_to_petri_net(self.powl)
        self.pnet = (net, im, fm)
        self.state.save_model((net, im, fm))

    def cc_alignments(self, net: PNet, im, fm) -> dict[float, float, float]:
        fitness = pm4py.conformance_diagnostics_alignments(self.event_log, net, im, fm)
        precision = pm4py.precision_alignments(self.event_log, net, im, fm)
        f1 = (
            2 * (fitness * precision) / (fitness + precision)
            if (fitness + precision) > 0
            else 0
        )

        self._add_context(
            f"Conformance Checking - Alignments: Fitness: {fitness}, Precision: {precision}, F1 Score: {f1}"
        )
        return (fitness, precision, f1)

    def cc_token_based_replay(self, net: PNet, im, fm) -> dict[float, float, float]:
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

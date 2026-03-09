import ast
import base64
import io
import os
import re

import pandas as pd
from PIL import Image, ImageChops

from promoai.general_utils import constants

from promoai.model_generation.code_extraction import execute_code_and_get_variable


def image_to_base64(file_pathway: str, max_dim=768) -> str:
    with Image.open(file_pathway) as img:
        # Greyscale and trim whitespaces
        img = img.convert("L")
        bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)
        # Ratio-based resizing to minimize token usage
        ratio = max_dim / float(max(img.size))
        if ratio < 1.0:
            new_size = tuple([int(x * ratio) for x in img.size])
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(
            buffer, format="JPEG", quality=60
        )  # Quality 60 is plenty for line graphs
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_dataframe_for_llms(file_pathway: str) -> str:

    if os.path.getsize(file_pathway) == 0:
        return "The artifact exists but contains no data."

    df = pd.read_csv(file_pathway)
    return df.to_markdown()


def transform_dataframe_for_llms(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "The artifact exists but contains no data."
    # return it as latex string encapsuled in ```latex ```
    if len(dataframe) < 50:
        return f"```markdown\n{dataframe.to_markdown(index = False)}\n```"
    else:
        statistics_per_column = {
            col: f"{dataframe[col].dtype}, {dataframe[col].nunique()} unique values"
            for col in dataframe.columns
        }
        # add mean, median, std for numeric columns
        numeric_cols = dataframe.select_dtypes(include="number").columns
        for col in numeric_cols:
            statistics_per_column[
                col
            ] += f", mean: {dataframe[col].mean():.2f}, median: {dataframe[col].median():.2f}, std: {dataframe[col].std():.2f}"
        stats_string = "\n".join(
            [f"- {col}: {stats}" for col, stats in statistics_per_column.items()]
        )
        return f"The dataframe has {len(dataframe)} rows and {len(dataframe.columns)} columns. Here are the statistics per column: \n{stats_string}"


def get_expected_event_log_format() -> str:
    return (
        "The expected event log format is a pandas DataFrame with at least the following columns:\n"
        "- 'case:concept:name': Identifier for each case (string)\n"
        "- 'concept:name': Name of the activity (string)\n"
        "- 'time:timestamp': Timestamp of the event (datetime)\n"
        "- 'org:resource': Resource associated with the event (string), this is optional\n\n"
        "Additional columns may be present, but these are the minimum required for process mining tasks."
    )


def _check_whitelisted_libraries(generated_code: str):
    whitelist = {
        "pandas",
        "pm4py",
        "promoai",
        "matplotlib",
        "seaborn",
        "numpy",
        "plotly",
        "datetime",
    }
    try:
        tree = ast.parse(generated_code)
    except SyntaxError as e:
        raise Exception(f"Generated code contains syntax errors: {e}")

    for node in ast.walk(tree):
        # import pandas as pd
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in whitelist:
                    raise Exception(
                        f"Generated code contains imports from non-whitelisted libraries: {top_level}"
                    )

        # from pm4py.objects.log import x
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise Exception(
                    "Generated code contains syntax errors: relative imports are not allowed"
                )

            top_level = node.module.split(".")[0]
            if top_level not in whitelist:
                raise Exception(
                    f"Generated code contains imports from non-whitelisted libraries: {top_level}"
                )


def _check_code_for_compilation(generated_code: str):
    try:
        compile(generated_code, "<string>", "exec")
    except SyntaxError as e:
        raise Exception(f"Generated code contains syntax errors: {e}")


def code_extraction_report(code_snippet: str, args, valid_artifact_ids: list):
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
    if constants.ENABLE_PRINTS:
        print(f"Extracted code:\n{code}")
    _check_whitelisted_libraries(code)
    _check_code_for_compilation(code)

    try:
        final_report = execute_code_and_get_variable(code, "final_report")
    except Exception as e:
        raise Exception(f"Error during code execution: {e}")
    if not isinstance(final_report, list):
        raise Exception(
            "The variable 'final_report' must be a list of dictionaries containing the report text and references to visualizations."
        )
    for entry in final_report:
        if entry["type"] == "artifact" and entry["content"] not in valid_artifact_ids:
            raise Exception(
                f"Invalid artifact reference in final_report: {entry['content']} is not a valid artifact key."
            )
        elif entry["type"] not in ["text", "artifact"]:
            raise Exception(
                f"Invalid type in final_report: {entry['type']}. Expected 'text' or 'artifact'."
            )

    return final_report, None

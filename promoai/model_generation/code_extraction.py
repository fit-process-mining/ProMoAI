import ast
import re
import traceback
from typing import Dict, Optional


def extract_final_python_code(response_text):
    python_code_pattern = r"```python(.*?)```"
    allowed_import_path = "promoai.model_generation.generator"
    allowed_import_class = "ModelGenerator"
    any_import_pattern = r"^\s*(from\s+\S+\s+import\s+\S+|import\s+\S+)"
    allowed_import_pattern = (
        r"^\s*(from\s+"
        + re.escape(allowed_import_path)
        + r"\s+import\s+"
        + re.escape(allowed_import_class)
        + r"|import\s+"
        + re.escape(allowed_import_path)
        + r"\."
        + re.escape(allowed_import_class)
        + r")\s*$"
    )

    matches = re.findall(python_code_pattern, response_text, re.DOTALL)

    if matches:
        python_snippet = matches[-1].strip()
        lines = python_snippet.split("\n")

        for line in lines:
            if re.match(any_import_pattern, line):
                if not re.match(allowed_import_pattern, line):
                    raise Exception(
                        "Python snippet does not meet the import statement requirements! Only the following import"
                        " statement is allowed: "
                        + "from "
                        + allowed_import_path
                        + " import "
                        + allowed_import_class
                    )
        return python_snippet

    else:
        raise Exception("No Python code snippet found!")


def execute_code_and_get_variable(
    code, variable_name, namespace: Optional[Dict] = None
):
    try:
        local_vars = (namespace or {}).copy()
        exec(code, globals(), local_vars)
        try:
            value = local_vars[variable_name]
        except KeyError:
            raise ValueError(f"Variable '{variable_name}' not found!")
        return value
    except Exception:
        exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
        error_msg = traceback.format_exception_only(exc_type, exc_value)[-1].strip()

        line_number, error_line = None, "Error line not directly available."
        filename = "<string>"

        for frame in traceback.extract_tb(exc_traceback):
            if frame.filename == filename:
                line_number = frame.lineno
                try:
                    error_line = code.split("\n")[line_number - 1]
                except IndexError:
                    error_line = "Line number out of range."
                break

        if line_number:
            error_details = f'Error occurred at line {line_number}: "{error_line}" with message: {error_msg}'
        else:
            error_details = (
                f"Error occurred with message: {error_msg}. \n The error occurred with trying to execute "
                f"the following extracted code: {code}. "
            )

        raise Exception(error_details)


def extract_resources_from_code(code):
    """
    Extract gen.activity calls with their pool and lane.

    Returns:
        dict: activity_name -> (pool, lane)
    """
    tree = ast.parse(code)
    resources = {}
    variables = {}

    class ActivityVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            # var = "value" or var = None
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                variables[var_name] = self.resolve_value(node.value)

            if isinstance(node.value, ast.Call):
                self.process_call(node.value)

            self.generic_visit(node)

        def visit_Expr(self, node):
            # Standalone expressions like: gen.activity("A", ...)
            if isinstance(node.value, ast.Call):
                self.process_call(node.value)
            self.generic_visit(node)

        def process_call(self, call):
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "activity":
                # Activity name
                activity_name = None
                if len(call.args) >= 1:
                    activity_name = self.resolve_value(call.args[0])
                if activity_name is None:
                    return
                # Positional args (if any)
                pool_val = None
                lane_val = None
                if len(call.args) >= 2:
                    pool_val = self.resolve_value(call.args[1])
                if len(call.args) >= 3:
                    lane_val = self.resolve_value(call.args[2])

                # Override with keyword args if these are provided
                for kw in call.keywords:
                    if kw.arg == "pool":
                        pool_val = self.resolve_value(kw.value)
                    elif kw.arg == "lane":
                        lane_val = self.resolve_value(kw.value)

                resources[activity_name] = (pool_val, lane_val)

        def resolve_value(self, node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return variables.get(node.id, None)
            return None

    ActivityVisitor().visit(tree)
    return resources

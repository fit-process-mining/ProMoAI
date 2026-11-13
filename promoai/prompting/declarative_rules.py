"""Prompt construction and parsing for declarative process constraints."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from powl.discovery.rule_based import (
    AbstractRule,
    AtMostOnceRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)

from promoai.general_utils.llm_connection import (
    LLMConnection,
    generate_result_with_error_handling,
)


_RULE_TYPES: dict[str, type[AbstractRule]] = {
    "AtMostOnce": AtMostOnceRule,
    "Init": InitializationRule,
    "Initialization": InitializationRule,
    "End": EndRule,
    "Existence": ExistenceRule,
    "CoExistence": CoExistenceRule,
    "NotCoExistence": NotCoExistenceRule,
    "NotSuccession": NotSuccessionRule,
    "Precedence": PrecedenceRule,
    "ChainPrecedence": ChainPrecedenceRule,
    "RespondedExistence": RespondedExistenceRule,
    "Response": ResponseRule,
    "ChainResponse": ChainResponseRule,
}
_RULE_TYPES.update({f"{name}Rule": cls for name, cls in tuple(_RULE_TYPES.items()) if name != "Init"})
_CANONICAL_NAMES = {cls: name for name, cls in _RULE_TYPES.items() if not name.endswith("Rule")}
_CANONICAL_NAMES[InitializationRule] = "Init"

_FEW_SHOT_EXAMPLES = """Examples:

Input: Every application process must start with registration. An application may only be approved after its documents have been reviewed. If an application is approved, a welcome package must eventually be sent. A background check and a fraud check must either both occur or both be absent. Rejected applications must never receive a welcome package.
Activities: 'Register Application', 'Review Documents', 'Approve Application', 'Reject Application', 'Send Welcome Package', 'Background Check', 'Fraud Check'
Output:
```python
rule1 = Init("Register Application")
rule2 = Precedence("Review Documents", "Approve Application")
rule3 = Response("Approve Application", "Send Welcome Package")
rule4 = CoExistence("Background Check", "Fraud Check")
rule5 = NotSuccession("Reject Application", "Send Welcome Package")
```

Input: Every purchasing process starts when a purchase request is submitted. Whenever payment occurs, the process instance must also include an invoice. A payment may only be made after the purchase request has been approved. If payment is made, a confirmation must eventually be sent. The process ends when the order is closed.
Activities: 'Submit Purchase Request', 'Approve Purchase Request', 'Reject Purchase Request', 'Make Payment', 'Create Invoice', 'Send Payment Confirmation', 'Close Order'
Output:
```python
rule1 = Init("Submit Purchase Request")
rule2 = RespondedExistence("Make Payment", "Create Invoice")
rule3 = Precedence("Approve Purchase Request", "Make Payment")
rule4 = Response("Make Payment", "Send Payment Confirmation")
rule5 = End("Close Order")
```

Input: A case-management process begins with case registration. A case cannot be closed unless it has previously been investigated. Whenever a case is escalated, a supervisor review must occur somewhere in the same process instance. If the supervisor approves the escalation, the case must eventually be reassigned. A case cannot be both cancelled and closed.
Activities: 'Register Case', 'Investigate Case', 'Escalate Case', 'Supervisor Review', 'Approve Escalation', 'Reassign Case', 'Cancel Case', 'Close Case'
Output:
```python
rule1 = Init("Register Case")
rule2 = Precedence("Investigate Case", "Close Case")
rule3 = RespondedExistence("Escalate Case", "Supervisor Review")
rule4 = Response("Approve Escalation", "Reassign Case")
rule5 = NotCoExistence("Cancel Case", "Close Case")
```

Input: A warehouse replenishment process begins by checking the current inventory level. If an automated alert is sent, an order must eventually be placed with a supplier. The same applies when a manual alert is sent. Automated and manual alerts cannot both occur in the same process instance. Stock may only be inspected after it has been received. Recording the stock must occur directly after the inspection. Placing the stock on shelves and placing it in storage are mutually exclusive. The process ends when inventory levels are updated.
Activities: 'Check Current Inventory Level', 'Send Automated Alert', 'Send Manual Alert', 'Place Supplier Order', 'Receive Stock', 'Inspect Stock', 'Record Stock', 'Place Stock on Shelves', 'Place Stock in Storage', 'Update Inventory Levels'
Output:
```python
rule1 = Init("Check Current Inventory Level")
rule2 = Response("Send Automated Alert", "Place Supplier Order")
rule3 = Response("Send Manual Alert", "Place Supplier Order")
rule4 = NotCoExistence("Send Automated Alert", "Send Manual Alert")
rule5 = Precedence("Receive Stock", "Inspect Stock")
rule6 = ChainResponse("Inspect Stock", "Record Stock")
rule7 = NotCoExistence("Place Stock on Shelves", "Place Stock in Storage")
rule8 = End("Update Inventory Levels")
```

Input: A support ticket must start with ticket creation. Ticket classification must occur directly after ticket creation. A ticket can only be assigned after it has been classified. If a ticket is assigned to a specialist, that specialist must eventually provide a resolution. A ticket cannot be reopened unless it has previously been closed. Whenever a ticket is reopened, it must eventually be assigned again.
Activities: 'Create Ticket', 'Classify Ticket', 'Assign Ticket', 'Assign to Specialist', 'Provide Resolution', 'Close Ticket', 'Reopen Ticket'
Output:
```python
rule1 = Init("Create Ticket")
rule2 = ChainResponse("Create Ticket", "Classify Ticket")
rule3 = Precedence("Classify Ticket", "Assign Ticket")
rule4 = Response("Assign to Specialist", "Provide Resolution")
rule5 = Precedence("Close Ticket", "Reopen Ticket")
rule6 = Response("Reopen Ticket", "Assign Ticket")
```
"""


def generate_declare_prompt(
    process_description: str, activities: Iterable[str], include_examples: bool = True
) -> str:
    """Ask the model for constraints over the exact event-log activity labels."""
    if not process_description or not process_description.strip():
        raise ValueError("A process description is required.")
    labels = tuple(dict.fromkeys(activities))
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Supply a nonempty list of activity labels.")

    instructions = (
        "You are an expert in process mining and DECLARE modeling. Translate the "
        "process description into declarative constraints. Only include constraints "
        "supported by the description; do not infer additional behavior.\n\n"
        f"Available activity labels (use these strings exactly): {', '.join(repr(a) for a in labels)}.\n\n"
        "Use only these templates:\n"
        "- AtMostOnce(A): A occurs at most once per trace.\n"
        "- Init(A): Every trace starts with A.\n"
        "- End(A): Every trace ends with A.\n"
        "- Existence(A): Every trace contains A.\n"
        "- CoExistence(A, B): A and B occur together or neither occurs.\n"
        "- NotCoExistence(A, B): A and B cannot both occur in a trace.\n"
        "- NotSuccession(A, B): B never occurs after A.\n"
        "- Precedence(A, B): B can occur only after a previous A.\n"
        "- ChainPrecedence(A, B): Every B is immediately preceded by A.\n"
        "- RespondedExistence(A, B): If A occurs, B must occur somewhere in the trace.\n"
        "- Response(A, B): Every occurrence of A must eventually be followed by B.\n"
        "- ChainResponse(A, B): Every occurrence of A is immediately followed by B.\n\n"
        "Use exactly the supplied activity labels, without renaming, merging, or "
        "inventing activities. Use valid Python string literals, escaping quotes "
        "inside labels. Output ONLY one Python code block with assignments such as:\n"
        "```python\nrule1 = Init(\"Activity A\")\n"
        "rule2 = Response(\"Activity A\", \"Activity B\")\n```\n"
        "Do not include imports, explanations, or other code."
    )
    examples = f"\n\n{_FEW_SHOT_EXAMPLES}" if include_examples else ""
    return f"{instructions}{examples}\n\nProcess description:\n{process_description}"


def feedback_prompt(feedback: str) -> str:
    return f"The user provided the following feedback: {feedback}. Refine the preceding rules accordingly."


def _code_from_response(response: str) -> str:
    match = re.search(r"```(?:python)?[ \t]*\r?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "```" in response:
        raise ValueError("The DECLARE code block is incomplete or incorrectly formatted.")
    return response.strip()


def parse_declare_rules(
    response: str, activities: Iterable[str] | None = None
) -> tuple[str, list[AbstractRule]]:
    """Parse rule assignments without executing model-generated Python code."""
    if not isinstance(response, str) or len(response.encode("utf-8")) > 128_000:
        raise ValueError("DECLARE response exceeds the source size limit.")
    code = _code_from_response(response)
    if not code:
        raise ValueError("No DECLARE rules were provided.")

    if isinstance(activities, (str, bytes)):
        raise ValueError("Activities must be a list of exact activity labels.")
    allowed = set(activities) if activities is not None else None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid DECLARE syntax on line {exc.lineno}: {exc.msg}.") from exc

    if len(tree.body) > 500 or sum(1 for _ in ast.walk(tree)) > 8_000:
        raise ValueError("DECLARE response exceeds the rule count or syntax size limit.")
    parsed: list[AbstractRule] = []
    variable_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Each rule must be assigned to a single variable.")
            name = statement.targets[0].id
            if name in variable_names:
                raise ValueError(f"Duplicate rule variable: {name}.")
            variable_names.add(name)
            call = statement.value
        elif isinstance(statement, ast.Expr):
            call = statement.value
        else:
            raise ValueError("Only DECLARE rule calls and assignments are allowed.")

        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Each statement must call a supported DECLARE template.")
        rule_name = call.func.id
        rule_type = _RULE_TYPES.get(rule_name)
        if rule_type is None:
            raise ValueError(f"Unsupported DECLARE rule: {rule_name}.")
        if call.keywords or len(call.args) not in (1, 2):
            raise ValueError(f"Invalid arguments for {rule_name}.")
        args = []
        for arg in call.args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                raise ValueError("Rule arguments must be literal activity names.")
            if allowed is not None and arg.value not in allowed:
                raise ValueError(f"Unknown activity {arg.value!r}; use an event-log activity label.")
            args.append(arg.value)
        try:
            parsed.append(rule_type(*args))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {rule_name} rule: {exc}") from exc

    rules = list(dict.fromkeys(parsed))
    if not rules:
        raise ValueError("No DECLARE rules were found.")
    normalized_code = "\n".join(
        f"rule{i} = {_CANONICAL_NAMES[type(rule)]}({', '.join(repr(a) for a in rule.args)})"
        for i, rule in enumerate(rules, start=1)
    )
    return normalized_code, rules


_ERROR_MESSAGE = (
    "Return only DECLARE assignments in a Python code block, for example "
    "```python\nrule1 = Init('A')\nrule2 = Response('A', 'B')\n```. "
    "Use only the given activity labels and supported templates. "
)


def _query_rules(conversation, activity_labels, llm_connection):
    def extract(response: str, _auto_duplicate: bool = False):
        return parse_declare_rules(response, activity_labels)

    code, rules, history = generate_result_with_error_handling(
        conversation,
        extraction_function=extract,
        api_key=llm_connection.api_key,
        llm_name=llm_connection.llm_name,
        ai_provider=llm_connection.ai_provider,
        llm_args=llm_connection.args or {},
        max_iterations=3,
        additional_iterations=2,
        standard_error_message=_ERROR_MESSAGE,
    )
    return rules, sum(message.get("type") == "error" for message in history), code


def query_llm_for_declare_rules(
    process_description: str,
    activities: Iterable[str],
    llm_connection: LLMConnection,
    *,
    include_examples: bool = True,
) -> tuple[list[AbstractRule], int, str]:
    """Return (rules, correction attempts, editable rule code)."""
    activity_labels = tuple(activities)
    prompt = generate_declare_prompt(process_description, activity_labels, include_examples)
    return _query_rules([{"role": "user", "content": prompt}], activity_labels, llm_connection)


def refine_declare_rules(
    process_description: str,
    activities: Iterable[str],
    previous_code: str,
    feedback: str,
    llm_connection: LLMConnection,
) -> tuple[list[AbstractRule], int, str]:
    """Revise the existing rule set according to user feedback."""
    if not feedback or not feedback.strip():
        raise ValueError("Rule feedback cannot be empty.")
    activity_labels = tuple(activities)
    code, _ = parse_declare_rules(previous_code, activity_labels)
    conversation = [
        {"role": "user", "content": generate_declare_prompt(process_description, activity_labels, False)},
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {
            "role": "user",
            "content": feedback_prompt(feedback)
            + " Return the complete revised rule set in one Python code block.",
        },
    ]
    return _query_rules(conversation, activity_labels, llm_connection)

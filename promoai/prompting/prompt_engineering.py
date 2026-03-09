import inspect
from typing import List, Optional

from promoai.prompting.shots import RESOURCE_AWARE_SHOTS, SHOTS

import_statement = "from promoai.model_generation.generator import ModelGenerator"

ERROR_MESSAGE_FOR_MODEL_GENERATION = """
Please update the model to fix the error. Make sure" \
                          f" to save the updated final model is the variable 'final_model'. """

STRICT_PROMPT = False


def add_role():
    res = (
        "Your role: you are an expert in process modeling,"
        " familiar with common"
        " process constructs such as exclusive choice, do-redo loops, and partial orders."
        " Your task is to analyze the textual description of a process and transform it into a process model in"
        " the POWL language. When generating a model, be as precise"
        " as possible and capture all details of the process in the model. "
    )
    if STRICT_PROMPT:
        res = (
            res
            + "Please create the process model strictly depending on the provided description, without using "
            "any domain knowledge you might have. You are not supposed to correct any information in the "
            "process, rather fully reply on the provided textual description.\n\n"
        )
    else:
        res = (
            res + "Also act as the process owner and use"
            " your expertise and familiarity with the"
            " process context to fill in any missing knowledge. \n\n"
        )
    return res


def add_knowledge(resource_aware_discovery=False):
    prompt = (
        "Use the following knowledge about the POWL process modeling language:\n"
        "A POWL model is a hierarchical model. POWL models are recursively generated"
        " by combining submodels into a new model either using a partial order"
        " or a decision graph. "
        " We define three types of POWL models. The first type of POWL models is the base case consisting of a"
        " single activity. For the second type"
        " of POWL models, we use a decision graph to combine"
        " multiple POWL models into a new model. A decision graph can model complex control flow, where exactly one path is taken at a time."
        " They can model decision patterns: cyclic behavior and exclusive choice structures."
        " The third type of POWL models is defined as a partial"
        " order over n >= 2 submodels. A partial order is binary relation that is irreflexive, transitive,"
        " and asymmetry. "
        "Provide the Python code that "
        "recursively generate a POWL model. Save the final model is the"
        " variable 'final_model'. Do not try to execute the code, just return it. Assume the class ModelGenerator"
        " is properly implemented and can be imported using the import statement:"
        f" {import_statement}. ModelGenerator provides the functions"
        " described below:\n"
    )
    prompt = (
        (
            prompt
            + " - activity(label) generates an activity. It takes 1 string arguments,"
            " which is the label of the activity.\n"
        )
        if not resource_aware_discovery
        else (
            prompt
            + " - activity(label, pool, lane) generates an activity. It takes 3 string arguments,"
            " which are the label of the activity, the pool name, and the lane name."
        )
    )
    prompt = (
        prompt
        + " - partial_order(dependencies) takes 1 argument, which is a list of tuples of submodels. These tuples"
        " set the nodes of the partial order and specify the"
        " edges of the partial order (i.e., the sequential dependencies). The"
        " transitive closure of the added dependencies should conform with the irreflexivity"
        " requirement of partial orders. We interpret unconnected nodes in a partial order to be"
        " concurrent and connections between nodes as sequential dependencies. Use a partial order"
        " with no edges (with the parameter 'dependencies' set to a list of tuples of size 1) to model pure"
        " concurrency/independency; i.e., to model the relation "
        " between sub models that can all be happens at the same time/in any order. However, note that all of them"
        " need to happen unlike the xor case in decision graphs. The main difference is that with xor case you model alternative"
        " paths (either path_1 or path_2), while with a partial order you model concurrent paths (you do both"
        " path_1 and path_2). The general assumption is partial orders is that nodes are concurrent; however, you can"
        " still add sequential dependencies between certain nodes (as tuples in the list for the parameter"
        " 'dependencies'). For example, this is"
        " the case in systems where you execute all subprocesses but one of them must be completed before"
        " starting another one. Assume we have 4 submodel A, B, C, D. partial_order(dependencies=[(A, B), (B, C), (C, D)]) "
        "models a sequence A -> B -> C -> D; partial_order(dependencies=[(A,), (B,), (C,), (D,)]) models full"
        " concurrency; partial_order(dependencies=[(A,B), (C,), (D,)]) models"
        " concurrency with the sequential dependency A -> B. Avoid using a partial"
        " order as a child of"
        " another partial order to ensure not leaving out any sequential dependencies. To resolve this,"
        " you can combine the two orders.\n"
        " - decision_graph(dependencies) takes 1 argument, which is a list of tuples of submodels. These tuples"
        " define the nodes and the directed edges of a decision graph, which models control flow.\n\n"
        " A decision graph is a directed graph with implicitly defined start and end points."
        " You can define start and end nodes by using None as a source or target in the tuples. For example, (None, A): A is a start node; (B, None): B is an end node. "
        " Each tuple in the `dependencies` list represents a sequential pair"
        " For example, a list of tuples `[(A, B), (A, C)]` will create the directed edges"
        " `A -> B` and `A -> C`."
        " This structure is used to model complex process flows, where exactly one path is taken at a time. This is the main difference"
        " from a partial order: decision graphs model exclusive paths, i.e., structures"
        " where exactly one path can be executed. Partial orders, on the other hand, are used to model concurrency."
        " Assume we have 4 submodels A, B, C, D."
        " - `decision_graph(dependencies=[(None, A), (A, B), (A, C), (B, D), (B, C), (C, None), (D, None)])` models the following four sequences (paths):"
        " A->C, A->B->C, A->B->D where exactly one of these will be executed."
        " A is the only start node and C and D are the possible end nodes."
        " Decision graphs can also model cyclic behavior. For example, "
        " `decision_graph(dependencies=[(None, A), (A, B), (B, A), (A, None)])` models a loop between A and B,"
        " where A is the start and end node."
        " Decision graphs can also model exclusive choice structures of n >= 2 sub-models. For example, "
        " `decision_graph(dependencies=[(None, A), (None, B), (A, None), (B, None)])` models an exclusive choice between A and B."
        "IMPORTANT: a decision graph can additionally express simple sequential dependencies, e.g., A is followed by B."
        "Therefore, there is no need to add additional partial order to express them."
        "IMPORTANT: we define three helper functions:\n"
        " - copy(model) for any powl model 'model', you can always call it to get a copy"
        " of the same model. This is useful to model cases where a subporcess or activity can be executed exactly"
        "twice (not really in a loop)."
        " - skip(submodel) takes 1 argument, which is a submodel. It makes the submodel optional/skippable.\n"
        " - self_loop(submodel) takes 1 argument, which is a submodel. It creates a self-loop on the submodel, i.e., the subprocess must be executed at least once but can be repeated multiple times afterwards.\n"
        "IMPORTANT: You can model skippable loops by combining these two functions. For example, "
        "`skip(self_loop(A))` models a skippable loop of A, where A can be executed any number of times, including zero."
    )
    return prompt


def add_knowledge_about_resources():
    return (
        "Additionally, consider the following knowledge about pools and lanes:\n"
        "An activity may be assigned to a pool and a lane. A pool is used to define the boundaries of that participant's process. It typically refers to an organization."
        "A lane is used to define a distinct role within a pool."
        "Lanes help to organize and categorize activities within a pool based on the roles or responsibilities of different participants."
        """**Important** Do not generate too specific pool names, use generic ones like "Customer", "Supplier", "System", "Department", etc.\n"""
        "The name of each pool and lane should be derived from the process description."
        "Avoid repeating the same pool name for different pools.\n"
        "Make sure that each lane belongs to only one pool.\n"
        "Different **departments or roles within the same organization** should be modeled as lanes within one single pool.\n"
        "Sometimes, pools and lanes cannot be identified from the process description. In that case, you can assign them to 'None'.\n"
        "**Important** If you have managed to identify at least one pool, you cannot use 'None' for other pools.\n"
        "This is valid for lanes as well.\n"
        "IMPORTANT: DO NOT assign names to pools and lanes using variables, e.g., university_pool = 'University'. Always use string literals IN THE FUNCTION CALLS, e.g., pool='University'.\n"
    )


def add_process_description(process_description):
    return "This is the process description: " + process_description


def negative_prompting():
    return (
        "Avoid common mistakes. "
        "First, ensure that the transitive closure of the generated partial orders"
        " do not violate irreflexivity. Verify that all optional/skippable and"
        " repeatable parts are modeled correctly. Also validate that the same submodel"
        " is not used multiple times (e.g., in dependency_graph then in partial_oder)! You have three ways for avoiding"
        " this depending on the case: (1)"
        " consider using decision_graphs to model cyclic behaviour; (2) if you instead want to create a second instance"
        " of the same submodel, consider creating a copy of it; (3) if none of these two cases apply, then"
        " your structure is not correct. Ensure that you correctly model loops and exclusive choices between larger complete"
        " alternative/loop paths (i.e., between full paths, not decision points). Finally, do not create partial"
        " orders as children of other partial orders. "
        " Instead, combine dependencies at the same hierarchical level to avoid nested partial orders."
        " Example of Correct Use of Partial Order:\n"
        "```python\n"
        "poset = partial_order(dependencies=[(A, B), (B, C)])\n"
        "```\n\n"
        "Example of Incorrect Use of Partial Order:\n"
        "```python\n"
        "poset_1 partial_order(dependencies=[(B, C)])\n"
        "poset_2 = partial_order(dependencies=[(A, poset)])\n"
        "```\n\n"
        " A decision graph can have loops:"
        " a cycle (e.g., `[(None, A), (A, B), (B, A), (A, None)]`) means that you always start with A, then you can go to B, then back to A, and so on."
        "However, the cycle must include at least one start node and one end node (can be the same node)."
        "When using decision graphs, **keep the model as flat as possible**"
        "Examples of unflattened model:"
        "```python\n"
        "activity_block = partial_order(dependencies=[(A, B)])\n"
        "decision_graph = `decision_graph(dependencies=[(None, C), (C, D), (C, E), (E, None)])`\n"
        "model = partial_order(dependencies(activity_block, decision_graph))`\n"
        "```\n\n"
        "Instead, do this:"
        "```python\n"
        "model = decision_graph(dependencies=[(None, A), (A, B), (B, C), (C, D), (C, E), (E, None)])\n"
        "```\n\n"
    )


def code_generation():
    return (
        "At the end of your response provide a single Python code snippet (i.e., staring with '```python') that"
        " contains the full final code. \n\n"
    )


def add_few_shots(resource_aware_discovery=False):
    res = (
        "Please use few-shots learning. These are few illustrating shots extended with common errors that you"
        " should avoid for each example:\n"
    )
    used_shots = RESOURCE_AWARE_SHOTS if resource_aware_discovery else SHOTS
    for i in range(len(used_shots)):
        description, model, errors = used_shots[i]
        full_source = inspect.getsource(model)
        source_lines = full_source.split("\n")
        content_lines = source_lines[1:-2] + ["\n"]
        content_as_string = "\n".join(line[4:] for line in content_lines)
        res = res + f"Process description for example {i + 1}:\n{description}\n"
        res = res + f"Process model for example {i + 1}:\n"
        res = res + f"```python\n{import_statement}\n{content_as_string}"
        res = res + f"```\n"
        res = res + f"Common errors to avoid for example {i + 1}:\n{errors}\n"
    return res + "\n"


def create_model_generation_prompt(
    process_description: str, resource_aware_discovery: bool
) -> str:
    prompt = add_role()
    if resource_aware_discovery:
        prompt = prompt + add_knowledge_about_resources()
    prompt = prompt + add_knowledge(resource_aware_discovery)
    prompt = prompt + negative_prompting()
    prompt = prompt + add_few_shots(resource_aware_discovery)
    prompt = prompt + code_generation()

    if process_description is not None:
        prompt = prompt + add_process_description(process_description)

    return prompt


def create_conversation(
    process_description: Optional[str], resource_aware_discovery: bool
) -> List[dict[str:str]]:
    prompt = create_model_generation_prompt(
        process_description, resource_aware_discovery=resource_aware_discovery
    )
    conversation = [{"role": "user", "content": f"{prompt}"}]
    return conversation


def update_conversation(
    conversation: List[dict[str:str]], feedback: str
) -> List[dict[str:str]]:
    update_prompt = (
        "Please update the model to fix it based on the provided feedback. Please make sure the returned"
        " model matches the initial process description, all previously provided feedback, and the new"
        "feedback comment as well. Make sure to save the updated final model is the variable"
        " 'final_model'. This is the new feedback comment:" + feedback
    )
    conversation.append({"role": "user", "content": f"{update_prompt}"})
    return conversation


def cut_conversation(
    conversation: List[dict[str:str]], pos: int
) -> List[dict[str:str]]:

    if pos < 0 or pos >= len(conversation):
        raise ValueError("Invalid position to cut the conversation!")

    cut_convo = conversation[: pos + 1]
    # Ensure the last message is from the user and is not feedback
    while cut_convo:
        if (
            cut_convo[-1]["role"] != "user"
            or cut_convo[-1].get("type", "msg") == "error"
        ):
            cut_convo.pop()
        else:
            break
    if not cut_convo:
        raise Exception("No valid user message found to cut the conversation!")
    return cut_convo


def model_self_improvement_prompt():
    return (
        "Thank you! The model was generated successfully! Could you further improve the model? "
        "Please critically evaluate the process model and improve it accordingly **only where genuinely beneficial**. "
        "Potential improvement steps might for instance include adding missing activities, managing additional "
        "exceptions, increasing concurrency in execution, or elevating choices to higher levels. If you find the "
        "model already optimized or see no significant areas for enhancement, it is "
        "perfectly acceptable to make minimal adjustments (e.g., relabeling some activities) or to return the "
        "same model without any changes."
    )


def model_self_improvement_prompt_short():
    return (
        "Thank you! The model was generated successfully! Could you further improve the model? "
        "Please critically evaluate the process model against the initial process description and improve it"
        " accordingly **only where genuinely beneficial**. If you see no significant areas for enhancement, it is "
        "perfectly acceptable to return the same model without any changes. Regardless of whether you improve the"
        " model or not, make sure to include a Python snippet that contains the model in your response."
    )


def description_self_improvement_prompt(descr: str):
    res = f"""
    You are provided with a process description. Your task is to optimize this description to make it richer and more
     detailed, while ensuring that all additions are relevant, accurate, and directly related to the original process.
     The goal is to make the description more comprehensive and suitable for process modeling purposes. """

    res = (
        res
        + f"""Possible areas for enhancement include:\n
    - **Detail Enhancement:** Add specific details that are missing but crucial for understanding the process flow. \n
    - **Clarity Improvement:** Clarify any ambiguous or vague statements to ensure that the description is clear and understandable.\n
    - **Explicit Process Constructs:** Rephrase parts of the description to explicitly incorporate constructs.
    For example, change 'X happens in most cases' to 'there is an exclusive choice between performing X or
     skipping it'.\n

    Please answer by only returning the improved process description without any additional text in your response. Do
     not define concrete activity labels yourself!"\n

    The process description:\n

    {descr}
    """
    )

    return res

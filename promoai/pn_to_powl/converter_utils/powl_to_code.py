from pm4py.objects.process_tree.obj import Operator
from powl.objects.BinaryRelation import BinaryRelation
from powl.objects.obj import (
    DecisionGraph,
    EndNode,
    OperatorPOWL,
    SilentTransition,
    StartNode,
    StrictPartialOrder,
    Transition,
)

from promoai.prompting.prompt_engineering import import_statement


def translate_powl_to_code(powl_obj):
    """
    Translates a POWL object from pm4py into code using ModelGenerator.

    Args:
        powl_obj: The POWL object to translate.

    Returns:
        A string containing the Python code that constructs the equivalent POWL model using ModelGenerator.
    """
    code_lines = [import_statement, "gen = ModelGenerator()"]

    var_counter = [0]

    def get_new_var_name():
        var_name = f"var_{var_counter[0]}"
        var_counter[0] += 1
        return var_name

    def process_powl(powl):
        if isinstance(powl, Transition):
            var_name = get_new_var_name()
            if isinstance(powl, SilentTransition):
                code_lines.append(f"{var_name} = None")
            else:
                label = powl.label
                if powl._organization is not None and powl._role is not None:
                    code_lines.append(
                        f"{var_name} = gen.activity('{label}', pool='{powl._organization}',  lane='{powl._role}')"
                    )
                else:
                    code_lines.append(f"{var_name} = gen.activity('{label}')")
            return var_name

        elif isinstance(powl, StartNode) or isinstance(powl, EndNode):
            return None

        elif isinstance(powl, OperatorPOWL):
            operator = powl.operator
            children = powl.children
            if operator == Operator.XOR:
                rel = BinaryRelation(children)
                graph = DecisionGraph(rel, children, children, False)
                graph = graph.reduce_silent_transitions()
                return process_powl(graph)
            elif operator == Operator.LOOP:
                rel = BinaryRelation(children)
                do = children[0]
                redo = children[1]
                rel.add_edge(do, redo)
                rel.add_edge(redo, do)
                graph = DecisionGraph(rel, [do], [do], False)
                graph = graph.reduce_silent_transitions()
                return process_powl(graph)
            else:
                raise Exception("Unknown operator! This should not be possible!")

        elif isinstance(powl, StrictPartialOrder) or isinstance(powl, DecisionGraph):
            nodes = powl.order.nodes
            if isinstance(powl, StrictPartialOrder):
                order = powl.order.get_transitive_reduction()
            elif isinstance(powl, DecisionGraph):
                order = powl.order
            else:
                raise Exception("Unknown POWL object! This should not be possible!")
            node_var_map = {node: process_powl(node) for node in nodes}
            dependencies = []
            nodes_in_edges = set()
            for source in nodes:
                for target in nodes:
                    source_var = node_var_map[source]
                    target_var = node_var_map[target]
                    if order.is_edge(source, target):
                        dependencies.append(f"({source_var}, {target_var})")
                        nodes_in_edges.update([source, target])

            if isinstance(powl, StrictPartialOrder):
                # Include nodes not in any edge as singleton tuples
                for node in nodes:
                    if node not in nodes_in_edges:
                        var = node_var_map[node]
                        dependencies.append(f"({var},)")

            dep_str = ", ".join(dependencies)
            var_name = get_new_var_name()
            code_lines.append(
                f"{var_name} = gen.partial_order(dependencies=[{dep_str}])"
            ) if isinstance(powl, StrictPartialOrder) else code_lines.append(
                f"{var_name} = gen.decision_graph(dependencies=[{dep_str}])"
            )
            return var_name

        else:
            raise Exception(
                f"Unknown POWL object {type(powl)}! This should not be possible!"
            )

    final_var = process_powl(powl_obj)
    code_lines.append(f"final_model = {final_var}")

    return "\n".join(code_lines)

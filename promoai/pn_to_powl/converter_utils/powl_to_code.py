"""Serialize native POWL models into the safe ModelGenerator language."""

from powl.objects.tagged_powl import Activity, ChoiceGraph, PartialOrder, TaggedPOWL

from promoai.prompting.prompt_engineering import import_statement


def translate_powl_to_code(powl_obj: TaggedPOWL) -> str:
    """Return model construction code preserving behavior, resources and metadata.

    Labels are serialized as Python literals, including quotes and newlines.
    Frequency tags supported by POWL JSON (optional and unbounded repetition)
    are preserved. Unsupported bounded frequency ranges fail explicitly instead
    of silently changing the process. Input models are never mutated.
    """
    code_lines = [import_statement, "gen = ModelGenerator()"]
    counter = 0
    ancestors: set[int] = set()

    def name():
        nonlocal counter
        variable = f"var_{counter}"
        counter += 1
        return variable

    def emit(node: TaggedPOWL) -> str:
        if id(node) in ancestors:
            raise ValueError("A POWL model cannot contain itself as a submodel.")
        if not isinstance(node, TaggedPOWL):
            raise TypeError(f"Expected a native POWL model, found {type(node).__name__}.")
        if node.min_freq not in (0, 1) or node.max_freq not in (1, None):
            raise ValueError("ModelGenerator code supports min_freq 0/1 and max_freq 1/None.")
        ancestors.add(id(node))
        if isinstance(node, Activity):
            variable = name()
            arguments = [repr(node.label)]
            if node.organization is not None:
                arguments.append(f"pool={node.organization!r}")
            if node.role is not None:
                arguments.append(f"lane={node.role!r}")
            code_lines.append(f"{variable} = gen.activity({', '.join(arguments)})")
        elif isinstance(node, (PartialOrder, ChoiceGraph)):
            children = node.children
            if not children:
                raise ValueError("Cannot serialize a composite model with no children.")
            variables = {child: emit(child) for child in children}
            dependencies = []
            if isinstance(node, PartialOrder):
                graph = node.transitive_reduction()
                for source in children:
                    for target in children:
                        if graph.has_edge(source, target):
                            dependencies.append(f"({variables[source]}, {variables[target]})")
                    if graph.degree(source) == 0:
                        dependencies.append(f"({variables[source]},)")
                method = "partial_order"
            else:
                node.validate_connectivity()
                starts, ends = node.start_nodes(), node.end_nodes()
                for child in children:
                    if child in starts:
                        dependencies.append(f"(None, {variables[child]})")
                for source in children:
                    for target in children:
                        if node.is_edge(source, target):
                            dependencies.append(f"({variables[source]}, {variables[target]})")
                for child in children:
                    if child in ends:
                        dependencies.append(f"({variables[child]}, None)")
                method = "decision_graph"
            variable = name()
            code_lines.append(f"{variable} = gen.{method}(dependencies=[{', '.join(dependencies)}])")
        else:
            raise TypeError(f"Unsupported POWL model: {type(node).__name__}.")
        if node.min_freq == 0:
            code_lines.append(f"{variable} = gen.skip({variable})")
        if node.max_freq is None:
            code_lines.append(f"{variable} = gen.self_loop({variable})")
        if node.attributes:
            # Validate the supported JSON annotations on a detached value.
            from promoai.model_generation.generator import ModelGenerator

            placeholder = Activity() if isinstance(node, Activity) else PartialOrder()
            ModelGenerator().annotate(placeholder, **node.attributes)
            kwargs = ", ".join(f"{key}={value!r}" for key, value in sorted(node.attributes.items()))
            code_lines.append(f"{variable} = gen.annotate({variable}, {kwargs})")
        ancestors.remove(id(node))
        return variable

    code_lines.append(f"final_model = {emit(powl_obj)}")
    return "\n".join(code_lines)

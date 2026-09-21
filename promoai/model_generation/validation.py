from typing import List as TList

from powl.objects.tagged_powl import Activity, PartialOrder, TaggedPOWL


def validate_resource_structure(powl: TaggedPOWL):
    identified_pools = set()
    identified_lanes = set()
    children_to_check = [powl]
    while children_to_check:
        current_node = children_to_check.pop()
        if hasattr(current_node, "organization"):
            identified_pools.add(current_node.pool)
        if hasattr(current_node, "role"):
            identified_lanes.add(current_node.lane)
        if hasattr(current_node, "nodes"):
            children_to_check.extend(list(current_node.nodes))
    if None in identified_pools and len(identified_pools) > 1:
        raise Exception(
            "Invalid resource structure: None pool used alongside identified pools."
        )
    if None in identified_lanes and len(identified_lanes) > 1:
        raise Exception(
            "Invalid resource structure: None lane used alongside identified lanes."
        )


def validate_partial_orders_with_missing_transitive_edges(powl: TaggedPOWL):
    if isinstance(powl, PartialOrder):
        try:
            powl.validate()
        except Exception:
            raise Exception("A partial order must be a DAG!")
    if hasattr(powl, "_g"):
        for child in powl._g.nodes:
            validate_partial_orders_with_missing_transitive_edges(child)


def validate_unique_transitions(
    powl: TaggedPOWL,
) -> TList[Activity]:
    def _find_duplicates(lst):
        seen_els = []
        duplicates = []
        for item in lst:
            if item not in seen_els:
                seen_els.append(item)
            else:
                if item not in duplicates:
                    duplicates.append(item)
        return duplicates

    def _collect_leaves(node: TaggedPOWL):
        if isinstance(node, Activity):
            return [node]

        elif hasattr(node, "_g"):
            leaves = []
            for child in node._g.nodes:
                leaves = leaves + _collect_leaves(child)
            return leaves
        else:
            raise Exception(
                "Unknown model type! The following model is not a transition and has no children: "
                + str(node)
            )

    transitions = _collect_leaves(powl)
    duplicate_transitions = _find_duplicates(transitions)
    if len(duplicate_transitions) > 0:
        raise Exception(
            "Duplicate transitions! Each of the following transitions occurs in multiple submodels: "
            + str(
                [
                    t.label if t.label else "silent transition"
                    for t in duplicate_transitions
                ]
            )
        )
    return transitions

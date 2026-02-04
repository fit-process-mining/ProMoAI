from pm4py import PetriNet
from powl import convert_to_petri_net
from powl.objects.obj import POWL

def convert_workflow_net_to_powl(net: PetriNet) -> POWL:
    """
    Convert a Petri net to a POWL model.

    Parameters:
    - net: PetriNet

    Returns:
    - POWL model
    """
    return convert_to_petri_net(net)

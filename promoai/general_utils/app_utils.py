from enum import Enum


class InputType(Enum):
    TEXT = "Text"
    MODEL = "Model"
    DATA = "Data"
    RULES = "Rules"


class ViewType(Enum):
    BPMN = "BPMN"
    POWL = "POWL"
    PETRI = "Petri Net"


DISCOVERY_HELP = "The event log will be used to generate a process model using the POWL miner (see https://doi.org/10.1016/j.is.2024.102493)."

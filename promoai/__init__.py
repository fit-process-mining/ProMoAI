from promoai.general_utils.ai_providers import AIProviders
from promoai.general_utils.app_utils import InputType

__name__ = "promoai"
__version__ = "2.2.2"


def generate_model_from_text(*args, **kwargs):
    from promoai.main import generate_model_from_text as _generate_model_from_text

    return _generate_model_from_text(*args, **kwargs)


def generate_model_from_event_log(*args, **kwargs):
    from promoai.main import (
        generate_model_from_event_log as _generate_model_from_event_log,
    )

    return _generate_model_from_event_log(*args, **kwargs)


def generate_model_from_petri_net(*args, **kwargs):
    from promoai.main import (
        generate_model_from_petri_net as _generate_model_from_petri_net,
    )

    return _generate_model_from_petri_net(*args, **kwargs)


def generate_model_from_bpmn(*args, **kwargs):
    from promoai.main import generate_model_from_bpmn as _generate_model_from_bpmn

    return _generate_model_from_bpmn(*args, **kwargs)


def query_bpmn(*args, **kwargs):
    from promoai.main import query_bpmn as _query_bpmn

    return _query_bpmn(*args, **kwargs)

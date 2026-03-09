import pm4py
from pm4py.util import constants
from powl import convert_to_bpmn, convert_to_petri_net, view as view_powl

from powl.objects.obj import POWL

from promoai.model_generation import code_extraction
from promoai.model_generation.model_generation import generate_model

from promoai.pn_to_powl.converter_utils.powl_to_code import translate_powl_to_code
from promoai.prompting import create_conversation, update_conversation


class LLMProcessModelGenerator:
    def __init__(self, process_model, conversation):
        self.process_model = process_model
        self.conversation = conversation

    @classmethod
    def from_description(
        cls,
        process_description: str,
        api_key: str,
        ai_model: str,
        ai_provider: str,
        resource_aware_discovery: bool = False,
        llm_args: dict = None,
    ):
        init_conversation = create_conversation(
            process_description, resource_aware_discovery=resource_aware_discovery
        )
        code, process_model, conversation = generate_model(
            init_conversation,
            api_key=api_key,
            llm_name=ai_model,
            ai_provider=ai_provider,
            llm_args=llm_args,
        )
        return cls(process_model, conversation)

    @classmethod
    def from_powl(cls, powl_model: POWL):
        init_conversation = create_conversation(None, resource_aware_discovery=False)
        conversation = list(init_conversation)
        conversation.append(
            {
                "role": "user",
                "content": "Instead of starting with a process description, I will give the code of"
                " the initial process model:\n\n" + translate_powl_to_code(powl_model),
            }
        )
        return cls(powl_model, conversation)

    def get_conversation(self):
        return self.conversation

    def get_code(self):
        return code_extraction.extract_final_python_code(
            self.conversation[-1]["content"]
        )

    def get_powl(self):
        return self.process_model

    def get_petri_net(self):
        return convert_to_petri_net(self.process_model)

    def get_bpmn(self):
        bpmn_model = convert_to_bpmn(self.process_model)
        return bpmn_model

    def update(
        self,
        feedback: str,
        api_key: str,
        ai_model: str,
        ai_provider: str,
        llm_args: dict = None,
    ):
        self.conversation = update_conversation(self.conversation, feedback)
        code, self.process_model, self.conversation = generate_model(
            conversation=self.conversation,
            api_key=api_key,
            llm_name=ai_model,
            ai_provider=ai_provider,
            llm_args=llm_args,
        )
        self.process_model = self.process_model.simplify()

    def view_bpmn(self, image_format: str = "svg"):
        bpmn_model = self.get_bpmn()
        pm4py.view_bpmn(bpmn_model, format=image_format)

    def view_petri_net(self, image_format: str = "svg"):
        net, im, fm = self.get_petri_net()
        pm4py.view_petri_net(net, im, fm, format=image_format)

    def view_powl(self, image_format: str = "svg"):
        view_powl(self.process_model)

    def export_bpmn(self, file_path: str, encoding: str = constants.DEFAULT_ENCODING):
        if not file_path.lower().endswith("bpmn"):
            raise Exception(
                "The provided file path does not have the '.bpmn' extension!"
            )
        bpmn_model = self.get_bpmn()
        from pm4py.objects.bpmn.exporter import exporter

        exporter.apply(bpmn_model, file_path, parameters={"encoding": encoding})

    def export_petri_net(
        self, file_path: str, encoding: str = constants.DEFAULT_ENCODING
    ):
        if not file_path.lower().endswith("pnml"):
            raise Exception(
                "The provided file path does not have the '.pnml' extension!"
            )
        net, im, fm = self.get_petri_net()
        from pm4py.objects.petri_net.exporter import exporter as petri_exporter

        petri_exporter.apply(
            net=net,
            initial_marking=im,
            final_marking=fm,
            output_filename=file_path,
            parameters={"encoding": encoding},
        )


# def initialize(process_description: str | None, api_key: str, llm_name: str, ai_provider: str,
#                powl_model: POWL = None, n_candidates: int = 1, debug: bool = False):
#     best_cand = None
#     exception = ""
#     for i in range(n_candidates):
#         try:
#             cand = LLMProcessModelGenerator(process_description=process_description, api_key=api_key,
#                                             llm_name=llm_name, ai_provider=ai_provider,
#                                             powl_model=powl_model)
#             if n_candidates > 1:
#                 raise Exception("Currently, there is no support for multiple candidate generation!")
#             else:
#                 best_cand = cand
#         except Exception as e:
#             exception = str(e)
#     if best_cand is None:
#         raise Exception(exception)
#     return best_cand
#
#
# def update(generator: LLMProcessModelGenerator, feedback: str, api_key: str, llm_name: str,
#            ai_provider: str, n_candidates: int = 1, debug: bool = False):
#     best_cand = None
#     exception = ""
#     for i in range(n_candidates):
#         try:
#             cand = generator if n_candidates == 1 else deepcopy(generator)
#             cand.update(feedback, api_key, llm_name, ai_provider)
#             if n_candidates > 1:
#                 raise Exception("Currently, there is no support for multiple candidate generation!")
#             else:
#                 best_cand = cand
#         except Exception as e:
#             exception = str(e)
#     if best_cand is None:
#         raise Exception(exception)
#     return best_cand

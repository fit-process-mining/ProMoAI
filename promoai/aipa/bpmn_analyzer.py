from promoai.aipa.conversation import (
    create_conversation,
    create_message,
    create_process_model_representation,
)
from promoai.general_utils.llm_connection import query_llm


class BPMNAnalyzer:
    def __init__(
        self,
        api_key: str,
        ai_model: str,
        ai_provider: str,
        initial_query: str,
        bpmn_xml_string: str = None,
        bpmn_json_string: str = None,
        selected_elements_json: str = None,
        model_abstraction="simplified_xml",
        enable_role_prompting=True,
        enable_natural_language_restriction=True,
        enable_chain_of_thought=False,
        enable_process_analysis=False,
        enable_knowledge_injection=False,
        enable_few_shots_learning=False,
        enable_negative_prompting=False,
        enable_examples=True,
    ):

        self.api_key = api_key
        self.ai_model = ai_model
        self.ai_provider = ai_provider
        self.model_abstraction = model_abstraction
        self._conversation = create_conversation(
            role="system",
            model_abstraction=model_abstraction,
            enable_role_prompting=enable_role_prompting,
            enable_natural_language_restriction=enable_natural_language_restriction,
            enable_chain_of_thought=enable_chain_of_thought,
            enable_process_analysis=enable_process_analysis,
            enable_knowledge_injection=enable_knowledge_injection,
            enable_few_shots_learning=enable_few_shots_learning,
            enable_negative_prompting=enable_negative_prompting,
            enable_examples=enable_examples,
        )

        self._conversation.append(
            create_process_model_representation(
                model_abstraction=model_abstraction,
                role="system",
                xml_string=bpmn_xml_string,
                json_abstraction=bpmn_json_string,
                selected_elements_json=selected_elements_json,
            )
        )

        self.last_response = None
        self.ask(initial_query)

    def get_last_response(self) -> str | None:
        return self.last_response

    def ask(self, query: str, selected_elements_json: str = None) -> str | None:

        if selected_elements_json:
            selected_elements_query = f"\n \n The user has now exclusively selected the following elements of the BPMN model (represented as a json): {selected_elements_json}"
            system_message = create_message(message=selected_elements_query, role="system", model_abstraction=self.model_abstraction)
            self._conversation.append(system_message)
        
        user_message_content = create_message(
            query, role="user", model_abstraction=self.model_abstraction
        )
        self._conversation.append(user_message_content)

        self.last_response = query_llm(
            conversation=self._conversation,
            api_key=self.api_key,
            llm_name=self.ai_model,
            ai_provider=self.ai_provider,
        )
        self._conversation.append({"role": "assistant", "content": self.last_response})

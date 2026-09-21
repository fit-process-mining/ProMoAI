from enum import Enum


class AIProviders(Enum):
    GOOGLE = "Google"
    OPENAI = "OpenAI"
    DEEPSEEK = "DeepSeek"
    ANTHROPIC = "Anthropic"
    DEEPINFRA = "Deepinfra"
    MISTRAL_AI = "Mistral AI"
    OPENROUTER = "OpenRouter"
    COHERE = "Cohere"
    GROK = "Grok"
    AZURE = "Azure"


AI_MODEL_DEFAULTS = {
    AIProviders.GOOGLE.value: "gemini-flash-latest",
    AIProviders.OPENAI.value: "gpt-5.6-terra",
    AIProviders.DEEPSEEK.value: "deepseek-reasoner",
    AIProviders.ANTHROPIC.value: "claude-sonnet-5",
    AIProviders.DEEPINFRA.value: "openai/gpt-oss-20b",
    AIProviders.MISTRAL_AI.value: "mistral-large-latest",
    AIProviders.OPENROUTER.value: "openai/gpt-oss-20b",
    AIProviders.COHERE.value: "command-a-plus-05-2026",
    AIProviders.GROK.value: "grok-4.6",
    AIProviders.AZURE.value: "very-secret-llm",
}

DEFAULT_AI_PROVIDER = AIProviders.GOOGLE.value

AI_HELP_DEFAULTS = {
    AIProviders.GOOGLE.value: "Enter a Google model name. You can get a **free Google API key** and check the latest models under: https://ai.google.dev/.",
    AIProviders.OPENAI.value: "Enter an OpenAI model name. You can get an OpenAI API key and check the latest models under: https://openai.com/pricing.",
    AIProviders.DEEPSEEK.value: "Enter a DeepSeek model name. You can get a DeepSeek API key and check the latest models under: https://api-docs.deepseek.com/.",
    AIProviders.ANTHROPIC.value: "Enter an Anthropic model name. You can get an Anthropic API key and check the latest models under: https://www.anthropic.com/api.",
    AIProviders.DEEPINFRA.value: "Enter a model name available through Deepinfra. DeepInfra supports popular open-source large language models like Meta's LLaMa and Mistral, and it also enables custom model deployment. You can get a Deepinfra API key and check the latest models under: https://deepinfra.com/models.",
    AIProviders.MISTRAL_AI.value: "Enter a Mistral AI model name. You can get a Mistral AI API key and check the latest models under: https://mistral.ai/.",
    AIProviders.OPENROUTER.value: "Enter an OpenRouter model name. You can get an OpenRouter API key and check the latest models under: https://openrouter.ai/models. Note that free models are available, but you may need to sign up for an account.",
    AIProviders.COHERE.value: "Enter a Cohere model name. You can get a Cohere API key and check the latest models under: https://cohere.com/docs/models. They further provide free trial keys.",
    AIProviders.GROK.value: "Enter a Grok model name. You can get a Grok API key and check the latest models under: https://x.ai/api.",
    AIProviders.AZURE.value: "Enter your Azure OpenAI API key, resource endpoint URL, deployment name, and API version.",
}

MAIN_HELP = (
    "Select the AI provider you'd like to use. Google offers the Gemini models,"
    " which you can **try for free**,"
    " while OpenAI provides GPT models. DeepInfra supports popular open-source large language models like"
    " Meta's LLaMa and also enables custom model deployment. You can also try the models provided by"
    " DeepSeek, Anthropic, Mistral AI, Grok, and Cohere."
    " OpenRouter is a platform that allows you to access various models from different providers."
    " For some of them, you may have free access, but you may need to sign up for an account."
)

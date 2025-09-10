from .LLMEnums import LLMENums
from .providers import OpenAIProvider, CoHereProvider
class LLMProviderFactory:
    def __init__(self, config:dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMENums.OPENAI.value:
            return OpenAIProvider(
                api_url=self.config.OPENAI_API_URL,
                api_key=self.config.OPENAI_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )
        elif provider == LLMENums.COHERE.value:
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )
        elif provider == LLMENums.GOOGLE_GENAI.value:
            from .providers import GoogleGenAIProvider
            return GoogleGenAIProvider(
                api_key=self.config.GOOGLE_GENAI_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )
        else:
            raise ValueError(f"Unknown provider type: {provider}")
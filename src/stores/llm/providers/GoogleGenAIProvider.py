import logging

from google import genai
from google.genai import types

from stores.llm.LLMInterface import LLMInterface

from ..LLMEnums import GoogleGenAIEnums


class GoogleGenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = genai.Client(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

        self.enums = GoogleGenAIEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Set GoogleGenAI generation model to {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(
            f"Set GoogleGenAI embedding model to {model_id} with size {embedding_size}"
        )

    def process_text(self, text: str):
        if len(text) > self.default_input_max_characters:
            self.logger.warning(
                f"Input text exceeds maximum character limit of {self.default_input_max_characters}. Truncating."
            )
            text = text[: self.default_input_max_characters]
        return text.strip()

    def generate_text(
        self,
        prompt: str,
        chat_history=[],
        system_prompt: str = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):

        if not self.client:
            self.logger.error("GoogleGenAI client is not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model is not set.")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens
            else self.default_generation_max_output_tokens
        )

        temperature = (
            temperature if temperature else self.default_generation_temperature
        )

        chat_history.append(self.construct_prompt(prompt, GoogleGenAIEnums.USER.value))

        response = self.client.models.generate_content(
            model=self.generation_model_id,
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )

        if not response or not response.text:
            self.logger.error("No response from GoogleGenAI.")
            return None

        print(response)
        generated_text = response.text
        return generated_text

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "parts": [{"text": prompt}]}

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("GoogleGenAI client is not initialized.")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set.")
            return None

        processed_text = self.process_text(text)

        response = self.client.models.embed_content(
            model=self.embedding_model_id, contents=processed_text
        )

        if not response or not response.embeddings or not response.embeddings[0]:
            self.logger.error("No embedding returned from GoogleGenAI.")
            return None

        embedding = response.embeddings[0].values
        if len(embedding) != self.embedding_size:

            self.logger.warning(
                f"Expected embedding size {self.embedding_size}, but got {len(embedding)}"
            )

        return embedding

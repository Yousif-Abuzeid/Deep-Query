import logging

import cohere

from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, DocumentTypeEnum
from typing import List,Union
class CoHereProvider(LLMInterface):
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

        self.client = cohere.Client(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

        self.enums = CohereEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Set CoHere generation model to {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(
            f"Set CoHere embedding model to {model_id} with size {embedding_size}"
        )

    def process_text(self, text: str):
        if len(text) > self.default_input_max_characters:
            self.logger.warning(
                f"Input text exceeds maximum character limit of {self.default_input_max_characters}. Truncating."
            )
            text = text[: self.default_input_max_characters]
        return text.strip()

    def generate_text(self, prompt: str, chat_history=[], system_prompt: str = None, max_output_tokens: int = None, temperature: float = None):


        if not self.client:
            self.logger.error("CoHere client is not initialized.")
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

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        if not response or not response.text:
            self.logger.error("No response from CoHere API.")
            return None
        return response.text

    def construct_prompt(self, prompt, role):
        return {"role": role, "text": prompt}
    
    def embed_text(self, document_type, text:Union[str,List[str]]):
        if not self.client:
            self.logger.error("CoHere client is not initialized.")
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set.")
            return None

        input_type = CohereEnums.DOCUMENT if document_type == "DOCUMENT" else CohereEnums.QUERY

        response = self.client.embed(
            input_type=input_type,
            model=self.embedding_model_id,
            embedding_types=['float'],
            texts=[self.process_text(t) for t in text],
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("No embedding returned from CoHere API.")
            return None
        return [f for f in response.embeddings.float]

    
    ## the following methods are just to comply with langchain expectations of an embedding model wrapper
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_text(texts, document_type=DocumentTypeEnum.DOCUMENT.value)
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text, document_type=DocumentTypeEnum.QUERY.value)

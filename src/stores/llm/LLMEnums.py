from enum import Enum


class LLMENums(str, Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"


class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

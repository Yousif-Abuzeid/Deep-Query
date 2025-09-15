from enum import Enum


class LLMENums(str, Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GOOGLE_GENAI = "GOOGLE_GENAI"


class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CohereEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class GoogleGenAIEnums(Enum):
    SYSTEM = "model"
    USER = "user"
    ASSISTANT = "model"

class DocumentTypeEnum(Enum):
    DOCUMENT = "DOCUMENT"
    QUERY = "QUERY"


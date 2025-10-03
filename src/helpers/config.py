from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import List
class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int


    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str


    GENERATION_BACKEND: str  # Options: "OPENAI", "COHERE", "GOOGLE_GENAI"
    EMBEDDING_BACKEND: str  # Options: "OPENAI", "COHERE" , "GOOGLE_GENAI"
    
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None
    OPENAI_API_KEY: str = None
    GOOGLE_GENAI_API_KEY: str = None

    GENERATION_MODEL_ID_LITERAL: List[str] = None
    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DEFAULT_MAX_CHARACTERS: int = None
    GENERATION_DEFAULT_MAX_TOKENS: int = None
    GENERATION_DEFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND_LITERAL: List[str] = None
    VECTOR_DB_BACKEND: str  # Options: "QDRANT"
    VECTOR_DB_PATH: str # Path for Qdrant DB
    VECTOR_DB_DISTANCE_METHOD: str  # Options: "COSINE", "DOT"
    VECTOR_DB_DISTANCE_METHOD_LITERAL: List[str] = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100  # Threshold to create index on pgvector vector column

    TAVILY_API_KEY: str = None
    DEFAULT_LANG: str = "en"  # Default language for document processing
    PRIMARY_LANG: str = "en"  # Primary language for template parsing
    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
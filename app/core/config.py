from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DB_URL: str
    PORT: int = 8005

    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    MAX_TOKENS: int = 8192
    EMBEDDING_DIM: int = 768

    QDRANT_URL: str
    QDRANT_COLLECTION: str = "pdf_knowledge_base"

    TOP_K_CHUNKS: int = 5
    SCORE_THRESHOLD: float = 0.6
    MAX_HISTORY_MESSAGES: int = 10
    
    GROQ_API_KEY: SecretStr
    GROQ_MODEL: str
    TEMPERATURE: float = 0.3

    REDIS_URL: str
    HISTORY_TTL: int = 3600

    KEY_PREFIX: str
    MAX_INPUTS_PER_BATCH: int = 32

    MAX_CONTEXT_CHARS: int = 12000

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_OVERFETCH: int = 4


settings = Settings()
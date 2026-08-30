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

    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET: str = "user-attachments"
    S3_REGION: str = "us-east-1"

    MAX_ATTACHMENT_SIZE_MB: int = 20
    ALLOWED_ATTACHMENT_TYPES: list[str] = [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/webp",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    ATTACHMENT_QDRANT_COLLECTION: str = "user_uploads"
    ATTACHMENT_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    ATTACHMENT_TOP_K_CHUNKS: int = 5
    ATTACHMENT_SCORE_THRESHOLD: float = 0.55

    ATTACHMENT_INLINE_MAX_CHARS: int = 12_000
    ATTACHMENT_INLINE_MAX_PDF_PAGES: int = 8
    ATTACHMENT_MAX_PDF_PAGES: int = 50

    ATTACHMENT_PREVIEW_URL_EXPIRES_SECONDS: int = 900

    CLAMAV_HOST: str = "clamav"
    CLAMAV_PORT: int = 3310
    
    ATTACHMENT_MALWARE_SCAN_ENABLED: bool = False

    ATTACHMENT_MAX_RETRIES: int = 5
    ATTACHMENT_INGEST_TIMEOUT_SECONDS: int = 180
    ATTACHMENT_QUEUE_REDIS_DB: int = 1

    ATTACHMENT_WAIT_TIMEOUT_SECONDS: int = 210
    ATTACHMENT_WAIT_POLL_INTERVAL_SECONDS: float = 2.0

    LLAMA_CLOUD_API_KEY: str

    INTERNAL_SERVICE_KEY: str

    PDF_INGESTION_SERVICE_BASE_URL: str

    PDF_INGESTION_SERVICE_TIMEOUT_SECONDS: float = 30.0

    AZURE_TRANSLATOR_KEY: str
    AZURE_TRANSLATOR_ENDPOINT: str
    AZURE_TRANSLATOR_REGION: str

    TRANSLATION_ENABLED: bool = True
    AZURE_TRANSLATOR_CHAR_LIMIT: int = 45000
    AZURE_TRANSLATOR_DETECT_MIN_SCORE: float = 0.7

    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_MAX_RESULTS: int = 5
    MCP_SERVER_URL: str = "http://localhost:8006/mcp"
    MCP_SERVER_AUTH_TOKEN: SecretStr | None = None
    MCP_CLIENT_TIMEOUT_SECONDS: float = 20.0
    MCP_TOOL_DISCOVERY_TIMEOUT_SECONDS: float = 10.0


settings = Settings()
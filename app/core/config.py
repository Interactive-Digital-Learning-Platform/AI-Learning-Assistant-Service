from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DB_URL")
PORT = int(os.getenv("PORT"))


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DEVICE= os.getenv("EMBEDDING_DEVICE")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")


TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE"))


REDIS_URL = os.getenv("REDIS_URL")
HISTORY_TTL = int(os.getenv("HISTORY_TTL"))

KEY_PREFIX = os.getenv("KEY_PREFIX")

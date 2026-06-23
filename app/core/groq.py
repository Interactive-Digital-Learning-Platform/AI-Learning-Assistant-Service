from langchain_groq import ChatGroq
from app.core.config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model = GROQ_MODEL,
    temperature= TEMPERATURE,
    max_tokens=1000,
    timeout=60,
    max_retries=2,
    streaming=True
)
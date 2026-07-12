from langchain_groq import ChatGroq
from app.core.config import settings

llm = ChatGroq(
    model = settings.GROQ_MODEL,
    api_key= settings.GROQ_API_KEY,
    temperature= settings.TEMPERATURE,
    max_tokens=1000,
    timeout=60,
    max_retries=2,
    streaming=True
)
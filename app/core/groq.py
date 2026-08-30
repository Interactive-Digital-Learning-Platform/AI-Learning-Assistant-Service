from langchain_groq import ChatGroq

from app.core.config import settings

llm = ChatGroq(
    model = settings.GROQ_MODEL,
    api_key= settings.GROQ_API_KEY,
    temperature= settings.TEMPERATURE,
    timeout=60,
    max_retries=2,
    streaming=True,
    reasoning_format="hidden"
)

utility_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
    timeout=60,
    max_retries=2,
    streaming=False,
    reasoning_format="hidden",
)
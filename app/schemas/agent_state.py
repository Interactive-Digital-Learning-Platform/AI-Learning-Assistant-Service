from typing import Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.services.retrieval_service import SearchResult


class QueryClassification(BaseModel):
    intent: Literal["general", "rag"] = Field(description="The intent of the user query")


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(description="Rewritten query for the user query")


class AgentState(TypedDict):
    conversation_id: str
    user_id: str

    user_message: str
    history: list[BaseMessage]

    intent: Literal["general", "rag"]

    rewritten_query: str
    retrieved_chunks: list
    context: list[SearchResult]

    response: str
    sources: list
    rag_used: bool

    error: Optional[str]

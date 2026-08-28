from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from app.services.retrieval_service import SearchResult


class QueryClassification(BaseModel):
    intent: Literal["general", "rag"] = Field(description="The intent of the user query")


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(description="A standalone search query preserving the meaning and important "
                "details of the user's latest question")


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

    inline_attachment_ids: list[str]
    has_attachments: bool
    attachment_pending: bool

    error: NotRequired[str]

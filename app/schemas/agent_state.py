from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator
from typing_extensions import NotRequired, TypedDict

from app.core.config import settings
from app.schemas.retrieval import RetrievalStatus
from app.services.retrieval_service import SearchResult


class QueryAnalysis(BaseModel):
    intent: Literal["general", "rag", "web_search", "generate_pdf"] = Field(
        description="The intent of the user query"
    )
    standalone_query: str = Field(
        default="",
        description="The latest message rewritten as a self-contained search query",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="One self-contained query per distinct part of a compound question",
    )
    subject: str | None = Field(
        default=None,
        description="The school subject the question belongs to, from the configured list",
    )

    @field_validator("standalone_query", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        return value or ""

    @field_validator("subject", mode="before")
    @classmethod
    def _known_subject(cls, value):
        if not isinstance(value, str):
            return None
        key = " ".join(value.split()).casefold()
        return next((s for s in settings.KB_SUBJECTS if s.casefold() == key), None)

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _none_to_list(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return value


class AgentState(TypedDict):
    conversation_id: str
    user_id: str

    user_message: str
    user_message_id: str
    history: list[BaseMessage]

    language: str
    source_language: str
    reply_language: str
    original_user_message: str
    translated_response: str
    translation_inbound_complete: bool
    translation_failed: bool

    intent: Literal["general", "rag", "web_search", "generate_pdf"]

    rewritten_query: str
    sub_queries: list[str]
    grade: int | None
    subject: str | None
    retrieval_scope: dict[str, int | str]
    retrieved_chunks: list
    context: list[SearchResult]
    retrieval_status: RetrievalStatus
    retrieval_best_score: float | None

    response: str
    sources: list
    citation_count: int
    citation_fallback: bool
    rag_used: bool
    web_search_used: bool

    documents: list
    document_status: str
    document_title: str

    inline_attachment_ids: list[str]
    attachment_names: dict[str, str]
    has_attachments: bool
    message_has_attachments: bool
    attachment_pending: bool
    unavailable_attachments: list[str]

    error: NotRequired[str]

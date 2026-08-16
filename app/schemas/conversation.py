from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.message import MessageRole, SourceCitation


class ConversationCreate(BaseModel):
    user_id: str = Field(..., description="ID of the user")
    filename: Optional[str] = Field(
        None,
        description="Scope chat to a specific PDF filename. "
        "If None, searches across all user documents.",
    )


class ConversationResponse(BaseModel):
    conversation_id: UUID
    user_id: str
    title: Optional[str]
    filename: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="ID of the conversation")
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., description="ID of the user sending the message")


class ChatResponse(BaseModel):
    message_id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sources: list[SourceCitation] = []
    created_at: datetime


class SSETokenEvent(BaseModel):
    """Streamed during generation — one per token."""

    type: str = "token"
    token: str


class SSEDoneEvent(BaseModel):
    """Sent once when generation is complete."""

    type: str = "done"
    message_id: str
    sources: list[SourceCitation] = []


class SSEErrorEvent(BaseModel):
    """Sent if an error occurs during streaming."""

    type: str = "error"
    error: str

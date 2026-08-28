import enum
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.schemas.attachment import AttachmentPreview


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceCitation(BaseModel):
    filename: str
    page: int
    score: float


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    sources: list[SourceCitation] = []
    attachments: list[AttachmentPreview] = []

    class Config:
        from_attributes = True


class MessageHistoryResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
    has_more: bool
    next_cursor: Optional[str] = None

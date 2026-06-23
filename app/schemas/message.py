import enum
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


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

    class Config:
        from_attributes = True


class MessageHistoryResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
    has_more: bool
    next_cursor: Optional[str] = None

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
    filename: str = ""
    page: int = 0
    score: float = 0.0
    title: Optional[str] = None
    url: Optional[str] = None
    provider: Optional[str] = None
    snippet: Optional[str] = None


class GeneratedDocument(BaseModel):
    document_id: str
    filename: str
    mime_type: str = "application/pdf"
    page_count: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[str] = None
    status: str = "completed"


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    sources: list[SourceCitation] = []
    documents: list[GeneratedDocument] = []
    attachments: list[AttachmentPreview] = []
    is_translated: bool = False
    translated_content: Optional[str] = None
    translation_failed: bool = False

    class Config:
        from_attributes = True


class MessageHistoryResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
    has_more: bool
    next_cursor: Optional[str] = None

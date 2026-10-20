from datetime import datetime

from pydantic import BaseModel

from app.schemas.message import MessageRole, SourceCitation


class ShareLinkRequest(BaseModel):
    user_id: str


class ShareLinkResponse(BaseModel):
    token: str
    shared_at: datetime
    message_count: int


class SharedAttachment(BaseModel):
    filename: str
    content_type: str


# Deliberately carries no conversation/message/attachment ids: the read endpoints
# for live conversations are keyed by those ids, so exposing them here would let
# a viewer read past the snapshot.
class SharedMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime
    sources: list[SourceCitation] = []
    attachments: list[SharedAttachment] = []
    is_translated: bool = False
    translated_content: str | None = None


class SharedChatResponse(BaseModel):
    title: str | None = None
    shared_at: datetime
    messages: list[SharedMessage]

import enum
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AttachmentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY_INLINE = "ready_inline"
    INDEXED = "indexed"
    FAILED = "failed"


class ProcessingStage(str, enum.Enum):
    SCANNING = "scanning"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"


class AttachmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    conversation_id: UUID
    filename: str
    content_type: str
    byte_size: int
    status: AttachmentStatus
    stage: Optional[ProcessingStage] = None
    error_message: Optional[str] = None
    created_at: datetime
    preview_url: Optional[str] = None
    

class AttachmentPreview(BaseModel):
    id: UUID
    filename: str
    content_type: str
    status: AttachmentStatus
    preview_url: Optional[str] = None

    
PENDING_STATUSES = (AttachmentStatus.UPLOADED, AttachmentStatus.PROCESSING)
GROUNDABLE_STATUSES = (AttachmentStatus.READY_INLINE, AttachmentStatus.INDEXED)
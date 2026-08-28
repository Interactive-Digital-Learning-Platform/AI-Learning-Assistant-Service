from typing import Literal, Optional

from pydantic import BaseModel


class WebhookError(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None

class PDFIngestionWebhookPayload(BaseModel):

    event_id: str
    job_id: str
    external_reference_id: str
    source_service: str
    status: Literal["done","failed"]

    collection: Optional[str] = None
    chunks_created: Optional[int] = None
    pages_processed: Optional[int] = None
    completed_at: Optional[str] = None

    error: Optional[WebhookError] = None
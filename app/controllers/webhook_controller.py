import logging
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attachment_model import Attachment
from app.schemas.attachment import AttachmentStatus
from app.schemas.pdf_ingestion_webhook import PDFIngestionWebhookPayload

logger = logging.getLogger(__name__)


async def handle_pdf_ingestion_webhook(
    session: AsyncSession,
    payload: PDFIngestionWebhookPayload
) -> None:

    attachment = await session.get(Attachment, payload.external_reference_id)

    if attachment is None:
        logger.warning(f"Webhook for unknown attachment_id={payload.external_reference_id}")
        return

    if payload.status == "done":
        attachment.status = AttachmentStatus.INDEXED
        attachment.chunk_count = payload.chunks_created
        attachment.extraction_method = "pdf_ingestion_service"
        attachment.indexed_at = datetime.now(UTC)
    else:
        attachment.status = AttachmentStatus.FAILED
        attachment.error_message = (payload.error.message if payload.error else "Ingestion Failed")


    await session.commit()
import asyncio
import logging

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.attachment_model import Attachment
from app.schemas.attachment import (
    PENDING_STATUSES,
    AttachmentResponse,
    AttachmentStatus,
)
from app.services.storage_service import StorageService

IMAGE_CONTENT_TYPES={"image/png", "image/jpeg", "image/webp"}

logger = logging.getLogger(__name__)

async def to_attachment_response(attachment: Attachment, storage_service: StorageService) -> AttachmentResponse:
    response = AttachmentResponse.model_validate(attachment)

    if attachment.content_type in IMAGE_CONTENT_TYPES:
        response.preview_url = await storage_service.get_preview_url(attachment.storage_key)

    return response


async def wait_for_attachments(
    attachment_ids: list[str],
    conversation_id: str,
) -> list[AttachmentStatus]:
    
    deadline = asyncio.get_event_loop().time() + settings.ATTACHMENT_WAIT_TIMEOUT_SECONDS

    if attachment_ids:
        stmt = select(Attachment.status).where(Attachment.id.in_(attachment_ids))
    else:
        stmt = select(Attachment.status).where(
            Attachment.conversation_id == conversation_id,
            Attachment.status.in_(PENDING_STATUSES),
        )

    while True:
        async with async_session_maker() as session:
            statuses = [row[0] for row in (await session.execute(stmt)).all()]

        if not statuses or not any(s in PENDING_STATUSES for s in statuses):
            return statuses

        if asyncio.get_event_loop().time() >= deadline:
            logger.warning(
                "Attachment wait timed out — conversation=%s ids=%s last=%s",
                conversation_id, attachment_ids, statuses,
            )
            return statuses

        await asyncio.sleep(settings.ATTACHMENT_WAIT_POLL_INTERVAL_SECONDS)
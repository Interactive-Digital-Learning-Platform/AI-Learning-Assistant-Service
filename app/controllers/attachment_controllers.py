import logging
import uuid

import magic
from fastapi import HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.attachment_model import Attachment
from app.schemas.attachment import AttachmentResponse, AttachmentStatus
from app.utils.attachment import to_attachment_response
from app.utils.conversation import get_conversation_or_404

logger = logging.getLogger(__name__)


async def upload_attachment(
    session: AsyncSession,
    request: Request,
    conversation_id: str,
    user_id: str,
    file: UploadFile
):

    conversation = await get_conversation_or_404(conversation_id, session)

    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    max_bytes = settings.MAX_ATTACHMENT_SIZE_MB * 1024 * 1024
    raw = await file.read(max_bytes + 1)

    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    sniffed_type = magic.from_buffer(raw, mime=True)

    if sniffed_type not in settings.ALLOWED_ATTACHMENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {sniffed_type}")

    storage_key = f"{conversation.id}/{uuid.uuid4()}-{file.filename}"

    storage_service = request.app.state.storage_service
    await storage_service.upload(storage_key, raw, sniffed_type)

    attachment = Attachment(
        conversation_id=conversation.id,
        user_id=user_id,
        filename=file.filename or "unnamed",
        content_type=sniffed_type,
        byte_size=len(raw),
        storage_key=storage_key,
        status=AttachmentStatus.UPLOADED
    )

    session.add(attachment)
    await session.commit()
    await session.refresh(attachment)

    arq_pool = request.app.state.arq_pool

    await arq_pool.enqueue_job(
        "process_attachment",
        str(attachment.id),
        _job_id=f"attachment:{attachment.id}"
    )


    return await to_attachment_response(attachment, storage_service)


async def get_attachment(
    session: AsyncSession,
    request: Request,
    conversation_id: str,
    attachment_id: str,
    user_id: str
) -> AttachmentResponse:
    
    conversation = await get_conversation_or_404(conversation_id, session)

    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    attachment = await session.get(Attachment, attachment_id)

    if attachment is None or str(attachment.conversation_id) != conversation_id:
        raise HTTPException(status_code=404, detail="Attachment not found")

    storage_service = request.app.state.storage_service
    return await to_attachment_response(attachment, storage_service)
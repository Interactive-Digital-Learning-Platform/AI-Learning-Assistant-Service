import logging

from app.core.database import async_session_maker
from app.models.attachment_model import Attachment
from app.schemas.attachment import AttachmentStatus
from app.schemas.retrieval import SearchResult
from app.services.attachment_ingestion_service import AttachmentIngestionService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class InlineAttachmentService:

    def __init__(
        self,
        storage_service: StorageService,
        ingestion_service: AttachmentIngestionService
    ):
        self.storage_service = storage_service
        self.ingestion_service = ingestion_service


    async def get_inline_text(self, attachment_id: str) -> SearchResult | None:

        async with async_session_maker() as session:
            attachment = await session.get(Attachment, attachment_id)

        if attachment is None:
            logger.warning("Inline attachment lookup miss for attachment_id=%s", attachment_id)
            return None

        if attachment.status != AttachmentStatus.READY_INLINE:
            return None

        content = await self.storage_service.download(attachment.storage_key)

        text, extraction_method = await self.ingestion_service.extract_text(
            filename=attachment.filename,
            content=content,
            content_type=attachment.content_type
        )

        if not text:
            return None

        return SearchResult(
            text=text,
            score=1.0,
            metadata={
                "filename": attachment.filename,
                "attachment_id": str(attachment.id),
                "source": extraction_method
            }
        )
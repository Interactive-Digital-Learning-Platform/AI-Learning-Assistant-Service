import logging
from asyncio import wait_for

from app.core.config import settings
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

        text = attachment.extracted_text
        extraction_method = attachment.extraction_method or "llama_parse"

        if text is None:
            text, extraction_method = await self._extract_and_store(attachment)

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


    async def _extract_and_store(self, attachment: Attachment) -> tuple[str, str]:
        logger.info(
            "No stored text for inline attachment_id=%s; extracting it once", attachment.id
        )

        content = await self.storage_service.download(attachment.storage_key)

        text, extraction_method = await wait_for(
            self.ingestion_service.extract_text(
                filename=attachment.filename,
                content=content,
                content_type=attachment.content_type
            ),
            timeout=settings.ATTACHMENT_INLINE_PARSE_TIMEOUT_SECONDS,
        )

        if text:
            async with async_session_maker() as session:
                stored = await session.get(Attachment, attachment.id)

                if stored is not None:
                    stored.extracted_text = text
                    stored.extraction_method = extraction_method
                    await session.commit()

        return text, extraction_method

import io
import logging
from asyncio import to_thread

from httpx import HTTPError
from pypdf import PdfReader
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.pdf_ingestion_client import PDFIngestionClient
from app.core.config import settings
from app.core.database import async_session_maker
from app.models.attachment_model import Attachment
from app.schemas.attachment import AttachmentStatus
from app.services.attachment_ingestion_service import AttachmentIngestionService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


def _count_pdf_pages(raw: bytes) -> int:
    return len(PdfReader(io.BytesIO(raw)).pages)


async def _scan_for_malware(clamd_client, raw: bytes) -> bool:

    result = await to_thread(clamd_client.instream, io.BytesIO(raw))
    status, _ = result.get("stream", ("ERROR", None))

    return status != "FOUND"


async def _submit_delegated(attachment: Attachment, raw, pdf_ingestion_client: PDFIngestionClient, session: AsyncSession) -> None:
    try:
        await pdf_ingestion_client.submit_document(
            content=raw,
            filename=attachment.filename,
            attachment_id=str(attachment.id),
            conversation_id=str(attachment.conversation_id),
            user_id=attachment.user_id
        )

    except HTTPError as e:
        logger.exception("Delegate submission failed for attachment_id=%s", attachment.id)
        attachment.status = AttachmentStatus.FAILED
        attachment.error_message = f"Failed to submit for indexing: {e}"
        await session.commit()
        return

    await session.commit()


async def _mark_failed(attachment_id: str, message: str) -> None:
    try:
        async with async_session_maker() as session:
            attachment = await session.get(Attachment, attachment_id)
            if attachment is None:
                return
            if attachment.status in (
                AttachmentStatus.READY_INLINE,
                AttachmentStatus.INDEXED,
            ):
                return
            attachment.status = AttachmentStatus.FAILED
            attachment.error_message = message
            await session.commit()
    except Exception:
        logger.exception(
            "Failed to mark attachment_id=%s as FAILED", attachment_id
        )


async def process_attachment(ctx: dict, attachment_id: str) -> None:
    try:
        await _process_attachment(ctx, attachment_id)
    except Exception:
        logger.exception(
            "process_attachment crashed for attachment_id=%s", attachment_id
        )
        await _mark_failed(attachment_id, "Attachment processing failed")
        raise


async def _process_attachment(ctx: dict, attachment_id: str) -> None:
    storage_service: StorageService = ctx["storage_service"]
    ingestion_service: AttachmentIngestionService = ctx["ingestion_service"]
    pdf_ingestion_client: PDFIngestionClient = ctx["pdf_ingestion_client"]
    clamd_client = ctx["clamd_client"]

    async with async_session_maker() as session:
        attachment = await session.get(Attachment, attachment_id)

        if attachment is None:
            logger.warning("process_attachment: no such attachment_id=%s", attachment_id)
            return

        attachment.status = AttachmentStatus.PROCESSING
        await session.commit()

        raw = await storage_service.download(attachment.storage_key)

        if settings.ATTACHMENT_MALWARE_SCAN_ENABLED:
            if not await _scan_for_malware(clamd_client, raw):
                attachment.status = AttachmentStatus.FAILED
                attachment.error_message = "File failed malware scan"
                await session.commit()
                return
        else:
            logger.warning(
                "Skipping malware scan for attachment_id=%s (scanning disabled)",
                attachment.id,
            )

        if attachment.content_type == "application/pdf":
            page_count = _count_pdf_pages(raw)

            if page_count > settings.ATTACHMENT_MAX_PDF_PAGES:
                attachment.status = AttachmentStatus.FAILED
                attachment.error_message = (
                    f"PDF has {page_count} pages, exceeds the "
                    f"{settings.ATTACHMENT_MAX_PDF_PAGES}-page limit for attachments"
                )
                await session.commit()
                return

            if page_count > settings.ATTACHMENT_INLINE_MAX_PDF_PAGES:
                await _submit_delegated(attachment, raw, pdf_ingestion_client, session)
                return

        text, extraction_method = await ingestion_service.extract_text(
            filename=attachment.filename,
            content=raw,
            content_type=attachment.content_type
        )

        if len(text) > settings.ATTACHMENT_INLINE_MAX_CHARS:
            logger.warning(
                "Extracted text for attachment_id=%s (%s) is %d chars, over "
                "ATTACHMENT_INLINE_MAX_CHARS (%d) — proceeding with READY_INLINE anyway",
                attachment.id, attachment.content_type, len(text),
                settings.ATTACHMENT_INLINE_MAX_CHARS,
            )

        attachment.status = AttachmentStatus.READY_INLINE
        attachment.extraction_method = extraction_method
        await session.commit()
        
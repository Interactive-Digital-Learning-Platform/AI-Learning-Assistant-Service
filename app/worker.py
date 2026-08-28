import logging

import httpx
from clamd import ClamdNetworkSocket

from app.clients.pdf_ingestion_client import PDFIngestionClient
from app.core.arq import get_arq_redis_settings
from app.core.config import settings
from app.jobs.attachment_jobs import process_attachment
from app.services.attachment_ingestion_service import AttachmentIngestionService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

async def startup(ctx: dict) -> None:
    ctx["storage_service"] = StorageService()
    ctx["ingestion_service"] = AttachmentIngestionService()
    ctx["clamd_client"] = (
        ClamdNetworkSocket(host=settings.CLAMAV_HOST, port=settings.CLAMAV_PORT)
        if settings.ATTACHMENT_MALWARE_SCAN_ENABLED
        else None
    )
    if not settings.ATTACHMENT_MALWARE_SCAN_ENABLED:
        logger.warning(
            "ATTACHMENT_MALWARE_SCAN_ENABLED=false — attachment malware scanning is DISABLED"
        )

    http_client = httpx.AsyncClient()
    ctx["http_client"] = http_client
    ctx["pdf_ingestion_client"] = PDFIngestionClient(http_client)

    logger.info("Attachment worker started")


async def shutdown(ctx: dict) -> None:
    await ctx["http_client"].aclose()
    logger.info("Attachment worker shut down")


class WorkerSettings:
    functions = [process_attachment]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = get_arq_redis_settings()

    max_tries = settings.ATTACHMENT_MAX_RETRIES
    job_timeout = settings.ATTACHMENT_INGEST_TIMEOUT_SECONDS
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers.webhook_controller import handle_pdf_ingestion_webhook
from app.core.database import get_async_session
from app.dependencies.require_internal_auth import require_internal_auth
from app.schemas.pdf_ingestion_webhook import PDFIngestionWebhookPayload

router = APIRouter(
    prefix="/internal/webhooks",
    tags=["Internal-Webhooks"],
    dependencies=[Depends(require_internal_auth)]
)


@router.post("/pdf-ingestion", status_code=204)
async def pdf_ingestion_webhook(
    payload: PDFIngestionWebhookPayload,
    session: Annotated[AsyncSession, Depends(get_async_session)]
):
    await handle_pdf_ingestion_webhook(session, payload)
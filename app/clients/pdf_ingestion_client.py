import json

import httpx

from app.core.config import settings


class PDFIngestionClient:

    def __init__(self, http_client: httpx.AsyncClient):
        self._http = http_client


    async def submit_document(
        self,
        *,
        content: bytes,
        filename: str,
        attachment_id: str,
        conversation_id: str,
        user_id: str
    ) -> str:


        response = await self._http.post(
            f"{settings.PDF_INGESTION_SERVICE_BASE_URL.rstrip('/')}/ingest/internal/documents",
            headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
            files={"file": (filename, content, "application/pdf")},
            data={
                "source_service": "ai-learning-assistant-service",
                "user_id": user_id,
                "external_reference_id": attachment_id,
                "qdrant_collection": settings.ATTACHMENT_QDRANT_COLLECTION,
                "metadata": json.dumps({"conversation_id": conversation_id, "user_id": user_id}),
                "max_pages": str(settings.ATTACHMENT_MAX_PDF_PAGES),
                "callback_requested": "true"
            },
            timeout=settings.PDF_INGESTION_SERVICE_TIMEOUT_SECONDS
        )

        response.raise_for_status()

        return response.json()["job_id"]

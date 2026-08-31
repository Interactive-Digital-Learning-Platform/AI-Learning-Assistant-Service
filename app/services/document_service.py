import json
import logging
from asyncio import wait_for

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        tool: BaseTool | None,
        *,
        timeout_seconds: float = 40.0,
        max_body_chars: int = 20_000,
    ):
        self._tool = tool
        self._timeout = timeout_seconds
        self._max_body_chars = max_body_chars

    @property
    def enabled(self) -> bool:
        return self._tool is not None

    async def generate(self, *, title: str, markdown_body: str) -> dict | None:
        if self._tool is None:
            return None

        tool_call = {
            "type": "tool_call",
            "name": self._tool.name,
            "args": {
                "title": title[:200],
                "markdown_body": markdown_body[: self._max_body_chars],
            },
            "id": "generate_pdf",
        }

        try:
            message = await wait_for(
                self._tool.ainvoke(tool_call), timeout=self._timeout
            )
        except Exception:
            logger.exception("PDF generation call failed; continuing without a document")
            return None

        if getattr(message, "status", None) == "error":
            logger.warning(
                "generate_pdf tool returned an error: %s", self._message_text(message)
            )
            return None

        payload = self._payload_from_message(message)
        if not payload:
            return None

        if payload.get("status") != "completed":
            logger.warning("generate_pdf returned status=%s", payload.get("status"))
            return {"status": payload.get("status") or "failed"}

        return {
            "document_id": payload.get("document_id"),
            "filename": payload.get("filename"),
            "mime_type": payload.get("mime_type", "application/pdf"),
            "page_count": payload.get("page_count"),
            "download_url": payload.get("download_url"),
            "expires_at": payload.get("expires_at"),
            "status": "completed",
        }

    def _payload_from_message(self, message) -> dict:
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict) and "status" in artifact:
            return artifact

        text = self._message_text(message)
        if text:
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                data = None

            if isinstance(data, dict):
                return data

        return {}

    @staticmethod
    def _message_text(message) -> str:
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)

        return ""

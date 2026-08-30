import json
import logging
from asyncio import wait_for

from langchain_core.tools import BaseTool

from app.schemas.retrieval import SearchResult

logger = logging.getLogger(__name__)


class WebSearchService:
    def __init__(
        self,
        tool: BaseTool | None,
        *,
        timeout_seconds: float = 20.0,
        max_results: int = 5,
    ):
        self._tool = tool
        self._timeout = timeout_seconds
        self._max_results = max_results

    @property
    def enabled(self) -> bool:
        return self._tool is not None

    async def search(self, query: str) -> list[SearchResult]:
        if self._tool is None:
            return []

        tool_call = {
            "type": "tool_call",
            "name": self._tool.name,
            "args": {"query": query, "max_results": self._max_results},
            "id": "web_search",
        }

        try:
            message = await wait_for(
                self._tool.ainvoke(tool_call), timeout=self._timeout
            )
        except Exception:
            logger.exception("Web search call failed; continuing without web results")
            return []

        if getattr(message, "status", None) == "error":
            logger.warning(
                "Web search tool returned an error: %s", self._message_text(message)
            )
            return []

        payload = self._payload_from_message(message)
        provider = payload.get("provider")

        results: list[SearchResult] = []
        for item in payload.get("results", []) or []:
            url = item.get("url")
            if not url:
                continue

            results.append(
                SearchResult(
                    text=item.get("content", "") or "",
                    score=float(item.get("score") or 0.0),
                    metadata={
                        "title": item.get("title") or url,
                        "url": url,
                        "source": provider,
                        "published_date": item.get("published_date"),
                    },
                )
            )

        logger.info(
            "Web search returned %d results (provider=%s)", len(results), provider
        )
        return results

    def _payload_from_message(self, message) -> dict:
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict) and "results" in artifact:
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

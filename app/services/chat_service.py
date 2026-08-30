import logging
from asyncio import wait_for
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.prompts.chat_prompts import (
    general_system_prompt,
    rag_system_prompt,
    web_search_system_prompt,
)
from app.prompts.graph_prompts import query_rewrite_prompt
from app.schemas.agent_state import RewrittenQuery
from app.services.retrieval_service import SearchResult

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, llm: BaseChatModel, utility_llm: BaseChatModel | None = None):
        self.llm = llm
        self._utility_llm = utility_llm or llm
        self.parser = StrOutputParser()

    def _prepare(
        self,
        message: str,
        history: list,
        context: list[SearchResult],
        mode: str,
        resolved_query: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        question = message

        if (
            resolved_query
            and resolved_query.strip()
            and resolved_query.strip().lower() != message.strip().lower()
        ):
            question = (
                f"{message}\n\n"
                f"(Resolved form of this question, for reference: {resolved_query})"
            )

        input_vars: dict[str, Any] = {"history": history, "question": question}

        if mode == "rag":
            input_vars["context"] = self.format_context(context)
            return rag_system_prompt, input_vars

        if mode == "web_search":
            input_vars["context"] = self.format_web_context(context)
            return web_search_system_prompt, input_vars

        return general_system_prompt, input_vars

    async def stream_response(
        self,
        message: str,
        history: list,
        context: list[SearchResult],
        mode: str
    ) -> AsyncGenerator[str, None]:

        prompt, input_vars = self._prepare(message, history, context, mode)

        chain = prompt | self.llm | self.parser

        logger.info(
            f"Generating response — "
            f"mode={mode} "
            f"chunks={len(context)} "
            f"history_msgs={len(history)}"
        )

        try:
            async for token in chain.astream(input_vars):
                yield token
        except Exception:
            logger.exception("LLM stream error")
            raise

    async def get_response(
        self,
        message: str,
        history: list,
        context: list[SearchResult],
        mode: str,
        resolved_query: str | None = None,
    ):

        prompt, input_vars = self._prepare(message, history, context, mode, resolved_query)

        chain = prompt | self.llm | self.parser

        logger.info(
            f"Generating response — "
            f"mode={mode} "
            f"chunks={len(context)} "
            f"history_msgs={len(history)}"
        )

        try:
            response = await wait_for(
                chain.ainvoke(input_vars),
                timeout=60.0
            )

            return response

        except Exception as e:
            logger.exception(f"LLM response generation error: {e}")
            raise
            

    @staticmethod
    def format_context(chunks: list[SearchResult]) -> str:
        if not chunks:
            return ""

        parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            filename = chunk.metadata.get("filename") or "the document"
            page = chunk.metadata.get("page_start")
            header = f"Excerpt {i} from {filename}"

            if isinstance(page, int) and page > 0:
                header += f", page {page}"

            part = f"{header}:\n{chunk.text}"

            parts.append(part)
            total_chars += len(part)

            if total_chars > settings.MAX_CONTEXT_CHARS:
                logger.warning(
                    f"Context truncated at chunk {i} — exceeded {settings.MAX_CONTEXT_CHARS} chars"
                )
                break

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def format_web_context(chunks: list[SearchResult]) -> str:
        if not chunks:
            return ""

        parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("title", "Untitled")
            url = chunk.metadata.get("url", "")
            published = chunk.metadata.get("published_date")
            header = f"Web result {i}: {title} ({url})"

            if published:
                header += f" — {published}"

            part = f"{header}\n{chunk.text}"

            parts.append(part)
            total_chars += len(part)

            if total_chars > settings.MAX_CONTEXT_CHARS:
                logger.warning(
                    f"Web context truncated at chunk {i} — exceeded {settings.MAX_CONTEXT_CHARS} chars"
                )
                break

        return "\n\n---\n\n".join(parts)

    async def rewrite_query(self, user_query: str, history: list[BaseMessage]):
        chain = query_rewrite_prompt | self._utility_llm | self.parser

        try:
            result = await wait_for(
                chain.ainvoke(
                    {"history": history or [], "user_query": user_query}
                ),
                timeout=15.0
            )

            rewritten = self._clean_rewrite(result)

            return RewrittenQuery(rewritten_query=rewritten or user_query)

        except TimeoutError:
            logger.warning(
                "Query rewriting timed out; using original user query"
            )

        except Exception:
            logger.exception(
                "Query rewriting failed; using original user query"
            )

        return RewrittenQuery(rewritten_query=user_query)

    @staticmethod
    def _clean_rewrite(raw: str | None) -> str:
        text = (raw or "").strip().strip('"').strip()

        if not text:
            return text

        prefixes = (
            "rewritten question:",
            "rewritten query:",
            "standalone question:",
            "standalone query:",
            "here is the rewritten question:",
            "here is the standalone question:",
            "question:",
            "query:",
        )

        lowered = text.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip().strip('"').strip()
                break

        return text.splitlines()[0].strip() if text else text

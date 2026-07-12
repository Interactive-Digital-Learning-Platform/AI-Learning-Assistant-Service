import logging
from asyncio import wait_for
from typing import Any, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser

from app.core.groq import llm
from app.prompts.chat_prompts import general_system_prompt, rag_system_prompt
from app.services.retrieval_service import SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.llm = llm
        self.parser = StrOutputParser()

    async def stream_response(
        self,
        message: str,
        history: list,
        context: list[SearchResult],
    ) -> AsyncGenerator[str, None]:

        rag_mode = len(context) > 0
        prompt = rag_system_prompt if rag_mode else general_system_prompt

        chain = prompt | self.llm | self.parser

        input_vars: dict[str, Any] = {"history": history, "question": message}

        if rag_mode:
            input_vars["context"] = self._format_context(context)

        logger.info(
            f"Generating response — "
            f"mode={'RAG' if rag_mode else 'general'} "
            f"chunks={len(context)} "
            f"history_msgs={len(history)}"
        )

        try:
            async for token in chain.astream(input_vars):
                yield token
        except Exception as e:
            logger.error(f"LLM stream error: {e}", exc_info=True)
            raise

    async def get_response(
        self, message: str, history: list, context: list[SearchResult]
    ) -> str:
        async def _collect() -> str:
            result = ""
            async for token in self.stream_response(message, history, context):
                result += token
            return result

        return await wait_for(_collect(), timeout=60.0)

    @staticmethod
    def _format_context(chunks: list[SearchResult]) -> str:
        if not chunks:
            return ""

        parts = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page_start", "?")
            part = f"[Source {i} — {filename}, Page {page}]\n{chunk.text}"

            parts.append(part)
            total_chars += len(part)

            if total_chars > settings.MAX_CONTEXT_CHARS:
                logger.warning(f"Context truncated at chunk {i} — exceeded {settings.MAX_CONTEXT_CHARS} chars")
                break
                
            parts.append(part)
        return "\n\n---\n\n".join(parts)

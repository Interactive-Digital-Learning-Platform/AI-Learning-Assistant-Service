import logging
from app.core.groq import llm
from langchain_core.output_parsers import StrOutputParser
from app.services.retrieval_service import SearchResult
from typing import AsyncGenerator
from app.prompts.chat_prompts import rag_system_prompt, general_system_prompt

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

        input_vars = {"history": history, "question": message}

        if rag_mode:
            input_vars["context"] = self._format_context(context)

        logger.info(
            f"Generating response — "
            f"mode={'RAG' if rag_mode else 'general'} "
            f"chunks={len(context)} "
            f"history_msgs={len(history)}"
        )

        async for token in chain.astream(input_vars):
            yield token

    async def get_response(
        self, message: str, history: list, context: list[SearchResult]
    ) -> str:
        full_response = ""
        async for token in self.stream_response(message, history, context):
            full_response += token
        return full_response

    @staticmethod
    def _format_context(chunks: list[SearchResult]) -> str:
        if not chunks:
            return ""

        parts = []

        for i, chunk in enumerate(chunks, 1):
            filename = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page_start", "?")
            parts.append(f"[Source {i} — {filename}, Page {page}]\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

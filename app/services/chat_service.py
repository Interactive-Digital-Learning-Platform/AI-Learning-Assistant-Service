import logging
from asyncio import wait_for
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.prompts.chat_prompts import (
    DOCUMENT_BASIS,
    NO_COURSE_MATERIAL,
    NO_WEB_RESULTS,
    RETRIEVAL_NOTICES,
    UNAVAILABLE_ATTACHMENTS_NOTICE,
    document_body_prompt,
    document_system_prompt,
    general_system_prompt,
    rag_system_prompt,
    web_search_system_prompt,
)
from app.services.retrieval_service import SearchResult
from app.utils.citations import CitationRewriter

logger = logging.getLogger(__name__)

_GROUNDED_MODES = ("rag", "web_search")
_MIN_EXCERPT_CHARS = 200


def is_user_file(chunk: SearchResult) -> bool:
    return "conversation_id" in chunk.metadata or "attachment_id" in chunk.metadata


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
        document_title: str = "",
        document_status: str = "",
        retrieval_status: str = "",
        unavailable_attachments: list[str] | None = None,
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

        if mode in _GROUNDED_MODES:
            input_vars["material"] = self._material_block(
                context, mode, retrieval_status, unavailable_attachments or []
            )
            prompt = rag_system_prompt if mode == "rag" else web_search_system_prompt
            return prompt, input_vars

        if mode == "generate_pdf":
            input_vars["document_title"] = document_title or "your document"
            input_vars["document_status"] = document_status or "completed"
            input_vars["document_basis"] = (
                DOCUMENT_BASIS.get(retrieval_status, "")
                if input_vars["document_status"] == "completed"
                else ""
            )
            return document_system_prompt, input_vars

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
        document_title: str = "",
        document_status: str = "",
        retrieval_status: str = "",
        unavailable_attachments: list[str] | None = None,
    ):

        prompt, input_vars = self._prepare(
            message,
            history,
            context,
            mode,
            resolved_query,
            document_title,
            document_status,
            retrieval_status,
            unavailable_attachments,
        )

        chain = prompt | self.llm | self.parser

        logger.info(
            f"Generating response — "
            f"mode={mode} "
            f"retrieval_status={retrieval_status or 'n/a'} "
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

    async def generate_document_body(
        self,
        topic: str,
        history: list,
        context: list[SearchResult],
        retrieval_status: str = "",
        unavailable_attachments: list[str] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        chain = document_body_prompt | self.llm | self.parser
        material = self._material_block(
            context, "generate_pdf", retrieval_status, unavailable_attachments or []
        )

        logger.info(
            f"Generating document body — "
            f"retrieval_status={retrieval_status or 'n/a'} "
            f"chunks={len(context)} "
            f"history_msgs={len(history or [])}"
        )

        raw_body = await wait_for(
            chain.ainvoke(
                {"history": history or [], "question": topic, "material": material}
            ),
            timeout=settings.DOCUMENT_BODY_TIMEOUT_SECONDS,
        )

        return self.finalize_citations(raw_body, context, "rag", visible=False)

    def finalize_citations(
        self,
        raw_response: str,
        context: list[SearchResult],
        mode: str,
        visible: bool | None = None,
    ) -> tuple[str, list[SearchResult]]:
        included = self.included_excerpts(context, mode)
        rewriter = CitationRewriter(
            visible=settings.CITATION_MARKERS_VISIBLE if visible is None else visible,
            excerpt_count=len(included),
        )
        text = rewriter.rewrite(raw_response)

        return text, [included[number - 1] for number in rewriter.order]

    @classmethod
    def included_excerpts(cls, context: list[SearchResult], mode: str) -> list[SearchResult]:
        if mode not in _GROUNDED_MODES or not context:
            return []

        user_files = [(i, c) for i, c in enumerate(context) if is_user_file(c)]
        course = [(i, c) for i, c in enumerate(context) if not is_user_file(c)]

        user_budget = (
            settings.ATTACHMENT_CONTEXT_CHARS if course else settings.MAX_CONTEXT_CHARS
        )

        fitted: dict[int, SearchResult] = {}
        user_used = cls._fit(user_files, user_budget, fitted)
        cls._fit(course, settings.MAX_CONTEXT_CHARS - user_used, fitted)

        if len(fitted) < len(context) or any(
            fitted[i].text != context[i].text for i in fitted
        ):
            logger.info(
                f"Context budget applied — kept {len(fitted)}/{len(context)} excerpts "
                f"(uploaded files {user_used} chars, budget {settings.MAX_CONTEXT_CHARS})"
            )

        return [fitted[i] for i in sorted(fitted)]

    @staticmethod
    def _fit(
        items: list[tuple[int, SearchResult]], budget: int, fitted: dict[int, SearchResult]
    ) -> int:
        used = 0

        for index, chunk in items:
            remaining = budget - used

            if len(chunk.text) <= remaining:
                fitted[index] = chunk
                used += len(chunk.text)
                continue

            if remaining >= _MIN_EXCERPT_CHARS:
                cut = chunk.text.rfind(" ", 0, remaining)
                cut = cut if cut > remaining // 2 else remaining
                text = chunk.text[:cut].rstrip() + " …"
                fitted[index] = replace(chunk, text=text)
                used += len(text)

            break

        return used

    def _material_block(
        self,
        context: list[SearchResult],
        mode: str,
        retrieval_status: str,
        unavailable_attachments: list[str],
    ) -> str:
        parts = self._excerpt_parts(context, "web_search" if mode == "web_search" else "rag")
        empty = NO_WEB_RESULTS if mode == "web_search" else NO_COURSE_MATERIAL
        blocks = ["\n\n---\n\n".join(parts) if parts else empty]

        notice = RETRIEVAL_NOTICES.get((mode, retrieval_status), "")
        if notice:
            blocks.append(notice)

        if unavailable_attachments:
            blocks.append(
                UNAVAILABLE_ATTACHMENTS_NOTICE.format(
                    filenames=", ".join(dict.fromkeys(unavailable_attachments))
                )
            )

        return "\n\n".join(blocks)

    @classmethod
    def _excerpt_parts(cls, chunks: list[SearchResult], mode: str) -> list[str]:
        label = cls._web_label if mode == "web_search" else cls._course_label

        return [
            f"{label(i, chunk)}\n{chunk.text}"
            for i, chunk in enumerate(cls.included_excerpts(chunks, mode), 1)
        ]

    @staticmethod
    def _course_label(index: int, chunk: SearchResult) -> str:
        metadata = chunk.metadata
        filename = metadata.get("filename") or "the document"

        if is_user_file(chunk):
            origin = f"Uploaded file: {filename}"
        else:
            origin = f"Textbook: {metadata.get('doc_title') or filename}"

        section_path = metadata.get("section_path")
        if isinstance(section_path, list) and section_path:
            origin += f" — {' › '.join(map(str, section_path))}"

        page = metadata.get("page_start")
        if isinstance(page, int) and page > 0:
            origin += f" (page {page})"

        return f"[S{index}] {origin}"

    @staticmethod
    def _web_label(index: int, chunk: SearchResult) -> str:
        title = chunk.metadata.get("title") or "Untitled"
        url = chunk.metadata.get("url", "")
        published = chunk.metadata.get("published_date")
        label = f"[S{index}] {title} ({url})"

        return f"{label} — {published}" if published else label

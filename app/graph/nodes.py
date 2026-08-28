import logging
from asyncio import gather, wait_for
from typing import Any, Awaitable, Literal

from langgraph.types import Command
from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.attachment_model import Attachment
from app.schemas.agent_state import AgentState
from app.schemas.attachment import (
    GROUNDABLE_STATUSES,
    PENDING_STATUSES,
    AttachmentStatus,
)
from app.schemas.retrieval import SearchResult
from app.services.chat_service import ChatService
from app.services.inline_attachment_service import InlineAttachmentService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService
from app.services.translator_service import TranslationError, TranslatorService
from app.utils.language import resolve_translation_plan

logger = logging.getLogger(__name__)


class GraphNodes:
    def __init__(
        self,
        session_service: SessionService,
        retrieval_service: RetrievalService,
        attachment_retrieval_service: RetrievalService,
        inline_attachment_service: InlineAttachmentService,
        chat_service: ChatService,
        translator_service: TranslatorService,
    ):
        self.session_service = session_service
        self.retrieval_service = retrieval_service
        self.attachment_retrieval_service = attachment_retrieval_service
        self.inline_attachment_service = inline_attachment_service
        self.chat_service = chat_service
        self.translator_service = translator_service


    async def ai_translator_node(self, state: AgentState) -> dict[str, Any]:
        if not state.get("translation_inbound_complete"):
            return await self._translate_inbound(state)

        return await self._translate_outbound(state)


    async def _translate_inbound(self, state: AgentState) -> dict[str, Any]:
        selected = state.get("language") or "English"
        original = state["user_message"]
        plan = resolve_translation_plan(selected, original)

        update: dict[str, Any] = {
            "original_user_message": original,
            "source_language": plan.source_language_hint,
            "reply_language": plan.reply_language,
            "translation_inbound_complete": True,
        }

        if plan.translate_inbound:
            result = await self.translator_service.to_english(original)
            update["user_message"] = result.text or original

            if result.detected_language:
                update["source_language"] = result.detected_language

            if (
                selected == "English"
                and result.detected_language == "en"
                and result.score >= settings.AZURE_TRANSLATOR_DETECT_MIN_SCORE
            ):
                update["reply_language"] = "English"

        return update


    async def _translate_outbound(self, state: AgentState) -> dict[str, Any]:
        if state.get("reply_language") != "Sinhala":
            return {}

        response = state.get("response") or ""

        if not response.strip():
            return {}

        try:
            translated = await self.translator_service.to_sinhala(response)
            return {"translated_response": translated}
        except TranslationError:
            logger.exception("Outbound translation failed; keeping English response")
            return {"translation_failed": True}


    async def _safe_search(self, search_coro: Awaitable[list[SearchResult]]) -> list[SearchResult]:
        try:
            return await wait_for(search_coro, timeout=15.0)
        except Exception:
            logger.exception("Document retrieval failed; continuing without this source")
            return []

            
    async def _load_inline_attachments(self, attachment_ids: list[str]) -> list[SearchResult]:

        if not attachment_ids:
            return []

        results = await gather(
            *(self.inline_attachment_service.get_inline_text(aid) for aid in attachment_ids),
            return_exceptions=True
        )

        inline_chunks: list[SearchResult] = []

        for attachment_id, result in zip(attachment_ids, results):
            if isinstance(result, BaseException):
                logger.exception(
                    "Inline attachment re-parse failed for attachment_id=%s", attachment_id
                )
                continue

            if result is not None:
                inline_chunks.append(result)

        return inline_chunks

        
    async def load_memory_node(
        self, state: AgentState
    ) -> Command[Literal["check_attachments"]]:
        logger.info("Load memory node is reached")
        history = await self.session_service.get_langchain_history(
            state["conversation_id"]
        )

        return Command(update={"history": history}, goto="check_attachments")

        
    async def check_attachments_node(self, state: AgentState) -> Command[Literal["classify_intent"]]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(Attachment.id, Attachment.status).where(
                    Attachment.conversation_id == state["conversation_id"]
                )
            )

            rows = result.all()

        inline_attachment_ids = [
            str(attachment_id) for attachment_id, status in rows
            if status == AttachmentStatus.READY_INLINE
        ]

        return Command(
            update={
                "inline_attachment_ids": inline_attachment_ids,
                "has_attachments": any(status in GROUNDABLE_STATUSES for _, status in rows),
                "attachment_pending": any(status in PENDING_STATUSES for _, status in rows)
            },
            goto="classify_intent"
        )
        
        
    async def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        response = await self.chat_service.rewrite_query(
            user_query=state["user_message"], history=state["history"]
        )

        logger.info(
            f"Query rewritten: '{state['user_message']}' -> '{response.rewritten_query}'"
        )

        return {"rewritten_query": response.rewritten_query}
        

    async def retrieve_docs_node(self, state: AgentState) -> dict[str, Any]:
        query = state["rewritten_query"] or state["user_message"]

        try:
            kb_chunks, attachment_chunks, inline_chunks = await gather(
                self._safe_search(self.retrieval_service.search(query=query)),
                self._safe_search(self.attachment_retrieval_service.search(
                    query=query,
                    filters={
                        "conversation_id": state["conversation_id"]
                    }
                )),
                self._load_inline_attachments(state["inline_attachment_ids"])
            )

            chunks = kb_chunks + attachment_chunks + inline_chunks

        except Exception:
            logger.exception(
                "Document retrieval failed; continuing without context"
            )
            chunks = []
            
        sources = [
            {
                "filename": c.metadata.get("filename", ""),
                "page": c.metadata.get("page_start", 0),
                "score": c.score,
            }
            for c in chunks
        ]

        return {
            "sources": sources,
            "context": chunks,
            "retrieved_chunks": chunks,
            "rag_used": True if state["intent"] == "rag" else False,
        }


    async def generate_response_node(self, state: AgentState) -> dict[str, Any]:
        logger.info("Response node is reached")
            
        response = await self.chat_service.get_response(
            message=state["rewritten_query"] or state["user_message"],
            history=state["history"],
            context=state["context"],
            mode=state["intent"]
        )

        return {"response": response}

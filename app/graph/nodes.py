import logging
from asyncio import gather, wait_for
from typing import Any, Awaitable, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command
from redis.exceptions import RedisError
from sqlalchemy import desc, select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.attachment_model import Attachment
from app.models.message_model import Message
from app.schemas.agent_state import AgentState
from app.schemas.attachment import (
    GROUNDABLE_STATUSES,
    PENDING_STATUSES,
    AttachmentStatus,
)
from app.schemas.retrieval import RetrievalOutcome, SearchResult
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.inline_attachment_service import InlineAttachmentService
from app.services.retrieval_service import (
    RetrievalService,
    is_transient_retrieval_error,
)
from app.services.session_service import SessionService
from app.services.translator_service import TranslationError, TranslatorService
from app.services.web_search_service import WebSearchService
from app.utils.citations import source_list
from app.utils.history import budget_history
from app.utils.language import resolve_translation_plan
from app.utils.rate_limit import TokenBucket

logger = logging.getLogger(__name__)


def _derive_document_title(body: str, *, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title[:200]
    return fallback[:200]


class GraphNodes:
    def __init__(
        self,
        session_service: SessionService,
        retrieval_service: RetrievalService,
        attachment_retrieval_service: RetrievalService,
        inline_attachment_service: InlineAttachmentService,
        chat_service: ChatService,
        translator_service: TranslatorService,
        web_search_service: WebSearchService,
        document_service: DocumentService,
    ):
        self.session_service = session_service
        self.retrieval_service = retrieval_service
        self.attachment_retrieval_service = attachment_retrieval_service
        self.inline_attachment_service = inline_attachment_service
        self.chat_service = chat_service
        self.translator_service = translator_service
        self.web_search_service = web_search_service
        self.document_service = document_service
        self._document_limiter = TokenBucket(settings.DOCUMENT_USER_RATE_LIMIT_PER_MIN)


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
            return await wait_for(search_coro, timeout=settings.RETRIEVAL_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("Document retrieval failed; continuing without this source")
            return []


    async def _retrieve_kb(
        self,
        query: str,
        sub_queries: list[str],
        top_k: int | None = None,
        grade: int | None = None,
        subject: str | None = None,
    ) -> RetrievalOutcome | Exception:
        if not settings.RETRIEVAL_SCOPE_ENABLED:
            grade, subject = None, None

        try:
            return await wait_for(
                self.retrieval_service.retrieve_scoped(
                    query=query,
                    grade=grade,
                    subject=subject,
                    top_k=top_k,
                    sub_queries=sub_queries,
                ),
                timeout=settings.RETRIEVAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            return exc


    def _resolve_kb_outcome(
        self, result: RetrievalOutcome | Exception, runtime: Runtime
    ) -> RetrievalOutcome:
        if not isinstance(result, Exception):
            return result

        attempt = (
            runtime.execution_info.node_attempt
            if runtime.execution_info
            else settings.RETRIEVAL_MAX_ATTEMPTS
        )

        if is_transient_retrieval_error(result) and attempt < settings.RETRIEVAL_MAX_ATTEMPTS:
            logger.warning(
                "Knowledge base retrieval failed on attempt %d (%s); retrying",
                attempt, type(result).__name__,
            )
            raise result

        logger.error(
            "Knowledge base retrieval failed; continuing without course material",
            exc_info=result,
        )
        return RetrievalOutcome(status="error")

            
    async def _load_inline_attachments(
        self, attachment_ids: list[str], names: dict[str, str]
    ) -> tuple[list[SearchResult], list[str]]:

        if not attachment_ids:
            return [], []

        results = await gather(
            *(self.inline_attachment_service.get_inline_text(aid) for aid in attachment_ids),
            return_exceptions=True
        )

        inline_chunks: list[SearchResult] = []
        unavailable: list[str] = []

        for attachment_id, result in zip(attachment_ids, results):
            if isinstance(result, BaseException):
                logger.error(
                    "Inline attachment text could not be loaded for attachment_id=%s",
                    attachment_id, exc_info=result,
                )
                unavailable.append(names.get(attachment_id, "your uploaded file"))
                continue

            if result is None:
                unavailable.append(names.get(attachment_id, "your uploaded file"))
                continue

            inline_chunks.append(result)

        return inline_chunks, unavailable

        
    async def _load_history_from_db(
        self, conversation_id: str, current_message_id: str
    ) -> list:
        query = select(Message).where(Message.conversation_id == conversation_id)

        if current_message_id:
            query = query.where(Message.id != current_message_id)

        async with async_session_maker() as session:
            result = await session.execute(
                query.order_by(desc(Message.created_at)).limit(
                    settings.MAX_HISTORY_MESSAGES * 2
                )
            )
            db_messages = list(reversed(result.scalars().all()))

        try:
            await self.session_service.warm_cache(conversation_id, db_messages)
        except RedisError:
            logger.exception(
                "History cache re-warm failed — conversation=%s", conversation_id
            )

        return self.session_service.history_from_db(db_messages)


    async def load_memory_node(
        self, state: AgentState
    ) -> Command[Literal["check_attachments"]]:
        logger.info("Load memory node is reached")
        conversation_id = state["conversation_id"]

        try:
            history = await self.session_service.get_cached_history(conversation_id)
        except RedisError:
            logger.exception(
                "History cache read failed; loading from the database — conversation=%s",
                conversation_id,
            )
            history = None

        if history is None:
            history = await self._load_history_from_db(
                conversation_id, state.get("user_message_id", "")
            )
            logger.info(
                "History loaded from the database — conversation=%s messages=%d",
                conversation_id, len(history),
            )

        budgeted = budget_history(history)

        if len(budgeted) < len(history) or any(
            b.content != h.content for b, h in zip(reversed(budgeted), reversed(history))
        ):
            logger.info(
                "History budgeted — conversation=%s messages=%d→%d chars=%d→%d",
                conversation_id, len(history), len(budgeted),
                sum(len(str(m.content)) for m in history),
                sum(len(str(m.content)) for m in budgeted),
            )

        return Command(update={"history": budgeted}, goto="check_attachments")

        
    async def check_attachments_node(self, state: AgentState) -> Command[Literal["classify_intent"]]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(
                    Attachment.id, Attachment.status, Attachment.filename, Attachment.message_id
                ).where(
                    Attachment.conversation_id == state["conversation_id"]
                )
            )

            rows = result.all()

        groundable = [row for row in rows if row.status in GROUNDABLE_STATUSES]
        current_message_id = state.get("user_message_id", "")

        return Command(
            update={
                "inline_attachment_ids": [
                    str(row.id) for row in groundable
                    if row.status == AttachmentStatus.READY_INLINE
                ],
                "attachment_names": {str(row.id): row.filename for row in groundable},
                "has_attachments": bool(groundable),
                "message_has_attachments": any(
                    str(row.message_id) == current_message_id for row in groundable
                ),
                "attachment_pending": any(row.status in PENDING_STATUSES for row in rows)
            },
            goto="classify_intent"
        )
        
        
    async def web_search_node(self, state: AgentState) -> dict[str, Any]:
        query = state["rewritten_query"] or state["user_message"]

        chunks = await self.web_search_service.search(query)

        logger.info("Web search node produced %d results", len(chunks))

        return {
            "context": chunks,
            "retrieved_chunks": chunks,
            "rag_used": False,
            "web_search_used": bool(chunks),
        }


    async def retrieve_docs_node(
        self, state: AgentState, runtime: Runtime
    ) -> Command[Literal["web_search", "generate_response", "generate_document"]]:
        query = state["rewritten_query"] or state["user_message"]

        sub_queries = state.get("sub_queries") or []
        for_document = state["intent"] == "generate_pdf"

        kb_result, attachment_chunks, (inline_chunks, unavailable) = await gather(
            self._retrieve_kb(
                query,
                sub_queries,
                settings.DOCUMENT_TOP_K_CHUNKS if for_document else None,
                state.get("grade"),
                state.get("subject"),
            ),
            self._safe_search(self.attachment_retrieval_service.search(
                query=query,
                filters={
                    "conversation_id": state["conversation_id"]
                },
                sub_queries=sub_queries,
            )),
            self._load_inline_attachments(
                state["inline_attachment_ids"], state.get("attachment_names", {})
            )
        )

        kb_outcome = self._resolve_kb_outcome(kb_result, runtime)

        user_chunks = inline_chunks + attachment_chunks
        chunks = (
            user_chunks + kb_outcome.results
            if state.get("message_has_attachments")
            else kb_outcome.results + user_chunks
        )

        retrieval_update = {
            "retrieval_status": kb_outcome.status,
            "retrieval_best_score": kb_outcome.best_score,
            "retrieval_scope": kb_outcome.scope,
            "unavailable_attachments": unavailable,
        }

        if (
            not for_document
            and kb_outcome.status == "none"
            and not chunks
            and not unavailable
            and self.web_search_service.enabled
        ):
            logger.info(
                "No relevant course material found (best_score=%s); falling back to web search",
                kb_outcome.best_score,
            )
            return Command(
                update={**retrieval_update, "intent": "web_search"}, goto="web_search"
            )

        return Command(
            update={
                **retrieval_update,
                "context": chunks,
                "retrieved_chunks": chunks,
                "rag_used": bool(chunks),
            },
            goto="generate_document" if for_document else "generate_response",
        )


    async def generate_document_node(self, state: AgentState) -> dict[str, Any]:
        topic = state["user_message"]
        user_id = state.get("user_id") or "default"

        if not self._document_limiter.allow(user_id):
            logger.warning("PDF generation rate limited for user_id=%s", user_id)
            return {
                "documents": [],
                "document_status": "rate_limited",
                "document_title": topic[:80],
            }

        context = state.get("context") or []

        try:
            body, cited = await self.chat_service.generate_document_body(
                topic,
                state["history"],
                context,
                retrieval_status=state.get("retrieval_status", "skipped"),
                unavailable_attachments=state.get("unavailable_attachments", []),
            )
        except Exception:
            logger.exception("Document body generation failed")
            return {
                "documents": [],
                "document_status": "failed",
                "document_title": topic[:80],
            }

        title = _derive_document_title(body, fallback=topic)

        result = await self.document_service.generate(title=title, markdown_body=body)

        if not result or result.get("status") != "completed":
            return {
                "documents": [],
                "document_status": "failed",
                "document_title": title,
            }

        included = self.chat_service.included_excerpts(context, "rag")
        citation_fallback = bool(included) and not cited

        if citation_fallback:
            logger.warning(
                "Document cited none of the %d excerpts; listing all of them as sources",
                len(included),
            )

        return {
            "documents": [result],
            "document_status": "completed",
            "document_title": result.get("filename") or title,
            "sources": source_list(included if citation_fallback else cited, "rag"),
            "citation_count": len(cited),
            "citation_fallback": citation_fallback,
        }


    async def generate_response_node(self, state: AgentState) -> dict[str, Any]:
        logger.info("Response node is reached")

        mode = state["intent"]
        context = state["context"]

        raw_response = await self.chat_service.get_response(
            message=state["user_message"],
            resolved_query=state["rewritten_query"],
            history=state["history"],
            context=context,
            mode=mode,
            retrieval_status=state.get("retrieval_status", "skipped"),
            unavailable_attachments=state.get("unavailable_attachments", []),
            document_title=state.get("document_title", ""),
            document_status=state.get("document_status", ""),
        )

        response, cited = self.chat_service.finalize_citations(raw_response, context, mode)

        if mode == "generate_pdf":
            return {"response": response}

        included = self.chat_service.included_excerpts(context, mode)
        citation_fallback = bool(included) and not cited

        if citation_fallback:
            logger.warning(
                "Answer cited none of the %d excerpts (mode=%s); listing all of them as sources",
                len(included), mode,
            )

        return {
            "response": response,
            "sources": source_list(included if citation_fallback else cited, mode),
            "citation_count": len(cited),
            "citation_fallback": citation_fallback,
        }

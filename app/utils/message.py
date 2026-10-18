import asyncio
import json
import logging
from datetime import UTC, datetime

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.attachment_model import Attachment
from app.models.conversation_model import Conversation
from app.models.message_model import Message, MessageRole
from app.schemas.agent_state import AgentState
from app.schemas.attachment import (
    GROUNDABLE_STATUSES,
    PENDING_STATUSES,
    AttachmentPreview,
)
from app.schemas.conversation import ChatRequest
from app.schemas.message import GeneratedDocument, MessageResponse, SourceCitation
from app.services.session_service import SessionService
from app.services.title_service import TitleService
from app.utils.attachment import wait_for_attachments
from app.utils.citations import CitationStreamFilter
from app.utils.language import resolve_translation_plan

logger = logging.getLogger(__name__)

_PDF_INTERIM_TOKEN = "Creating your PDF…\n\n"


def message_to_response(msg: Message, attachments: list[AttachmentPreview] | None = None) -> MessageResponse:
    meta = msg.message_metadata or {}
    sources = [
        SourceCitation(
            filename=s.get("filename", ""),
            page=s.get("page", 0),
            score=s.get("score", 0.0),
            title=s.get("title"),
            url=s.get("url"),
            provider=s.get("provider"),
            snippet=s.get("snippet"),
        )
        for s in meta.get("sources", [])
    ]
    documents = [
        GeneratedDocument(
            document_id=d.get("document_id", ""),
            filename=d.get("filename", ""),
            mime_type=d.get("mime_type", "application/pdf"),
            page_count=d.get("page_count"),
            download_url=d.get("download_url"),
            expires_at=d.get("expires_at"),
            status=d.get("status", "completed"),
        )
        for d in meta.get("documents", [])
    ]
    return MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
        sources=sources,
        documents=documents,
        attachments=attachments or [],
        is_translated=bool(msg.is_translated),
        translated_content=msg.translated_content,
    )

async def link_attachments_to_message(
    session: AsyncSession,
    attachment_ids: list[str],
    conversation_id,
    message_id
) -> None:

    if not attachment_ids:
        return

    result = await session.execute(
        select(Attachment).where(Attachment.id.in_(attachment_ids))
    )

    for attachment in result.scalars().all():
        if str(attachment.conversation_id) != str(conversation_id):
            logger.warning(
                "Skipping attachment_id=%s — belongs to a different conversation", attachment.id
            )
            continue

        if attachment.message_id is not None:
            logger.warning(
                "Skipping attachment_id=%s — already linked to a message", attachment.id
            )
            continue

        attachment.message_id = message_id
        

async def chat_stream_handler(
    conv: Conversation,
    request: ChatRequest,
    session: AsyncSession,
    assistant_graph: CompiledStateGraph,
    session_service: SessionService,
    title_service: TitleService,
    is_new_conversation: bool = False
):

    full_response = ""
    final_state = None

    conversation_id = str(conv.id)
    user_id = str(conv.user_id)
    committed = False

    plan = resolve_translation_plan(request.language, request.message)
    suppress_stream = plan.translate_outbound

    try:
        if is_new_conversation:
            yield {
                "data": json.dumps({
                    "type": "conversation_created",
                    "conversation_id": conversation_id
                })
            }
            
        user_message = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=request.message,
            token_count=int(len(request.message.split()) * 1.3),
        )
        session.add(user_message)
        await session.flush()

        await link_attachments_to_message(
            session, request.attachment_ids, conv.id, user_message.id
        )

        await session.commit()

        final_statuses = await wait_for_attachments(
            request.attachment_ids, conversation_id
        )

        if final_statuses:
            still_pending = any(s in PENDING_STATUSES for s in final_statuses)
            any_groundable = any(s in GROUNDABLE_STATUSES for s in final_statuses)

            if still_pending:
                yield {
                    "data": json.dumps({
                        "type": "error",
                        "error": "Your file is taking longer than expected to process. "
                                 "Please try sending your message again in a moment.",
                    })
                }
                return

            if not any_groundable:
                yield {
                    "data": json.dumps({
                        "type": "error",
                        "error": "I couldn't process the file you attached. Please try re-uploading it."
                    })
                }
                return

        initial_state: AgentState = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_message": request.message,
            "user_message_id": str(user_message.id),
            "history": [],
            "language": request.language,
            "source_language": "",
            "reply_language": plan.reply_language,
            "original_user_message": "",
            "translated_response": "",
            "translation_inbound_complete": False,
            "translation_failed": False,
            "intent": "general",
            "retrieved_chunks": [],
            "sources": [],
            "citation_count": 0,
            "citation_fallback": False,
            "rewritten_query": "",
            "sub_queries": [],
            "grade": request.grade,
            "subject": None,
            "retrieval_scope": {},
            "context": [],
            "retrieval_status": "skipped",
            "retrieval_best_score": None,
            "response": "",
            "rag_used": False,
            "web_search_used": False,
            "documents": [],
            "document_status": "",
            "document_title": "",
            "inline_attachment_ids": [],
            "attachment_names": {},
            "has_attachments": False,
            "message_has_attachments": False,
            "attachment_pending": False,
            "unavailable_attachments": [],
            "error": ""
        }

        stream = await assistant_graph.astream_events(
            initial_state,
            version="v3"
        )

        interim_sent = False

        async with asyncio.timeout(120.0):
            async for message in stream.messages:
                logger.info(f"Message object: {message}")

                if message.node == "generate_document":
                    if not interim_sent and not suppress_stream:
                        interim_sent = True
                        yield {
                            "data": json.dumps({
                                "type": "token",
                                "token": _PDF_INTERIM_TOKEN,
                            })
                        }

                    async for _ in message.text:
                        pass

                    continue

                if message.node != "generate_response":
                    continue

                citation_filter = CitationStreamFilter(
                    visible=settings.CITATION_MARKERS_VISIBLE
                )

                async for token in message.text:
                    if token:
                        full_response += token

                        if suppress_stream:
                            continue

                        visible_text = citation_filter.feed(token)

                        if visible_text:
                            yield {
                                "data": json.dumps({
                                    "type": "token",
                                    "token": visible_text,
                                })
                            }

                remaining = citation_filter.flush()

                if remaining and not suppress_stream:
                    yield {
                        "data": json.dumps({
                            "type": "token",
                            "token": remaining,
                        })
                    }

        final_state = await stream.output()

        if final_state is None:
            raise RuntimeError("Graph finished without producing a final state")

        english_response = final_state.get("response") or full_response
        translation_failed = final_state.get("translation_failed", False)
        english_user_message = final_state.get("user_message") or request.message
        translated_reply = final_state.get("translated_response") or ""

        if suppress_stream:
            display_text = translated_reply or english_response
            if display_text:
                yield {
                    "data": json.dumps({
                        "type": "token",
                        "token": display_text,
                    })
                }
        elif not full_response and english_response:
            full_response = english_response
            yield {
                "data": json.dumps({
                    "type": "token",
                    "token": english_response,
                })
            }

        sources = final_state.get("sources", [])
        documents = final_state.get("documents", [])

        if english_user_message != request.message:
            user_message.content = english_user_message
            user_message.translated_content = request.message
            user_message.is_translated = True

        if not conv.title:
            conv.title = await title_service.generate(
                english_user_message,
                english_response,
                reply_language=final_state.get("reply_language", "English"),
            )

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=english_response,
            token_count=int(len(full_response.split()) * 1.3),
            message_metadata={
                "sources": sources,
                "documents": documents,
                "intent": final_state.get("intent"),
                "rag_used": final_state.get("rag_used", False),
                "web_search_used": final_state.get("web_search_used", False),
                "document_status": final_state.get("document_status", ""),
                "rewritten_query": final_state.get("rewritten_query", ""),
                "retrieval_status": final_state.get("retrieval_status", "skipped"),
                "retrieval_best_score": final_state.get("retrieval_best_score"),
                "retrieval_scope": final_state.get("retrieval_scope", {}),
                "citation_count": final_state.get("citation_count", 0),
                "citation_fallback": final_state.get("citation_fallback", False),
            },
        )

        if final_state.get("reply_language") == "Sinhala" and translated_reply:
            assistant_msg.translated_content = translated_reply
            assistant_msg.is_translated = True

        session.add(assistant_msg)
        conv.updated_at = datetime.now(UTC)

        await session.commit()
        committed = True
        await session.refresh(assistant_msg)

        try:
            await session_service.append_message(
                conversation_id,
                MessageRole.USER,
                english_user_message
            )

            await session_service.append_message(
                conversation_id,
                MessageRole.ASSISTANT,
                english_response
            )

        except Exception as e:
            logger.exception(
                "Redis append failed — conversation=%s",
                conversation_id,
            )
            
        yield {
            "data": json.dumps(
                {
                    "type": "done",
                    "message_id": str(assistant_msg.id),
                    "sources": sources,
                    "documents": documents,
                    "translation_failed": translation_failed,
                    "conversation_title": conv.title,
                }
            )
        }

        logger.info(
            f"Message complete — conversation={conversation_id} "
            f"rag={final_state.get('rag_used', False)} "
            f"intent={final_state.get('intent')} "
            f"retrieval_status={final_state.get('retrieval_status')} "
            f"best_score={final_state.get('retrieval_best_score')} "
            f"citations={final_state.get('citation_count', 0)} "
            f"citation_fallback={final_state.get('citation_fallback', False)} "
            f"chunks={len(final_state.get('retrieved_chunks', []))}"
        )

    except asyncio.TimeoutError:
        logger.error(
            "Graph timeout — conversation=%s",
            conversation_id,
        )

        if not committed:
            await session.rollback()


        yield {
            "data": json.dumps({
                "type": "error",
                "error": "Request timed out. Please try again."
            })
        }

    except Exception as e:
        logger.error(
            "Chat stream failed — conversation=%s error=%s",
            conversation_id,
            e,
            exc_info=True,
        )

        if not committed:
            await session.rollback()
            
        yield {
            "data": json.dumps(
                {
                    "type": "error",
                    "error": "An error occurred while generating the response.",
                }
            )
        }

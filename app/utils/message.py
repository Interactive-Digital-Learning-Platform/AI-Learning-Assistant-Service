import asyncio
import json
import logging
from datetime import UTC, datetime

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.schemas.message import MessageResponse, SourceCitation
from app.services.session_service import SessionService
from app.utils.attachment import wait_for_attachments
from app.utils.language import resolve_translation_plan

logger = logging.getLogger(__name__)


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
    return MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
        sources=sources,
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
            "rewritten_query": "",
            "context": [],
            "response": "",
            "rag_used": False,
            "web_search_used": False,
            "inline_attachment_ids": [],
            "has_attachments": False,
            "attachment_pending": False,
            "error": ""
        }

        stream = await assistant_graph.astream_events(
            initial_state,
            version="v3"
        )

        async with asyncio.timeout(90.0):
            async for message in stream.messages:
                logger.info(f"Message object: {message}")
    
                if message.node != "generate_response":
                    continue
    
                async for token in message.text:
                    if token:
                        full_response += token

                        if suppress_stream:
                            continue

                        yield {
                            "data": json.dumps({
                                "type": "token",
                                "token": token,
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

        if english_user_message != request.message:
            user_message.content = english_user_message
            user_message.translated_content = request.message
            user_message.is_translated = True

        if not conv.title:
            conv.title = (
                english_user_message[:97] + "..."
                if len(english_user_message) > 100
                else english_user_message
            )


        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=english_response,
            token_count=int(len(full_response.split()) * 1.3),
            message_metadata={
                "sources": sources,
                "intent": final_state.get("intent"),
                "rag_used": final_state.get("rag_used", False),
                "web_search_used": final_state.get("web_search_used", False),
                "rewritten_query": final_state.get("rewritten_query", ""),
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
                    "translation_failed": translation_failed,
                }
            )
        }

        logger.info(
            f"Message complete — conversation={conversation_id} "
            f"rag={final_state.get('rag_used', False)} "
            f"intent={final_state.get('intent')}"
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

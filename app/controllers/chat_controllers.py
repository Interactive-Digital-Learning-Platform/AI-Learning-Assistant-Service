import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import HTTPException, Request
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.attachment_model import Attachment
from app.models.conversation_model import Conversation
from app.models.message_model import Message, MessageRole
from app.schemas.attachment import AttachmentPreview
from app.schemas.conversation import (
    ChatRequest,
    ConversationCreate,
)
from app.schemas.message import MessageHistoryResponse
from app.services.retrieval_service import RetrievalService
from app.services.storage_service import StorageService
from app.services.translator_service import TranslationError, TranslatorService
from app.utils.attachment import IMAGE_CONTENT_TYPES
from app.utils.conversation import conversation_to_response, get_conversation_or_404
from app.utils.language import resolve_translation_plan
from app.utils.message import (
    chat_stream_handler,
    link_attachments_to_message,
    message_to_response,
)

logger = logging.getLogger(__name__)


async def create_conversation(session: AsyncSession, data: ConversationCreate):
    try:
        conversation = Conversation(
            user_id=data.user_id,
        )

        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)

        return conversation_to_response(conversation, message_count=0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def list_conversations(
    session: AsyncSession, user_id: str, limit: int, offset: int
):

    try:
        result = await session.execute(
            select(Conversation, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(Conversation.id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
            .offset(offset)
        )

        rows = result.all()

        return [
            conversation_to_response(conv, message_count=count) for conv, count in rows
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_conversation(session: AsyncSession, conversation_id: str):
    try:
        conv = await get_conversation_or_404(conversation_id, session)
        count_result = await session.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        )
        count = count_result.scalar() or 0
        return conversation_to_response(conv, message_count=count)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def delete_conversation(
    session: AsyncSession, request: Request, conversation_id: str
):
    try:
        conv = await get_conversation_or_404(conversation_id, session)

        session_service = request.app.state.session_service
        await session_service.clear(conversation_id)

        attachment_result = await session.execute(
            select(Attachment.storage_key).where(Attachment.conversation_id == conv.id)
        )

        storage_keys = [row[0] for row in attachment_result.all()]

        attachment_retrieval_service: RetrievalService = request.app.state.attachment_retrieval_service
        storage_service: StorageService = request.app.state.storage_service

        await attachment_retrieval_service.delete_by_filter(
            filters={
                "conversation_id": conversation_id
            }
        )

        for storage_key in storage_keys:
            await storage_service.delete(storage_key)
        
        await session.delete(conv)
        await session.commit()
        logger.info(f"Conversation deleted — id={conversation_id}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_messages(
    session: AsyncSession,
    conversation_id: str,
    request: Request,
    limit: int,
    before: Optional[str],
):
    try:
        conv = await get_conversation_or_404(conversation_id, session)

        query = select(Message).where(Message.conversation_id == conv.id)

        if before:
            try:
                cursor_dt = datetime.fromisoformat(before)
                query = query.where(Message.created_at < cursor_dt)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid 'before' cursor — must be ISO timestamp",
                )

        messages_result = await session.execute(
            query.order_by(desc(Message.created_at)).limit(limit + 1)
        )
        messages = messages_result.scalars().all()

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        messages = list(reversed(messages))

        message_ids = [m.id for m in messages]
        attachments_by_message: dict = {}

        if message_ids:
            storage_service: StorageService = request.app.state.storage_service
            attachment_result = await session.execute(
                select(Attachment).where(Attachment.message_id.in_(message_ids))
            )

            for attachment in attachment_result.scalars().all():
                preview_url = None

                if attachment.content_type in IMAGE_CONTENT_TYPES:
                    preview_url = await storage_service.get_preview_url(attachment.storage_key)

                attachments_by_message.setdefault(attachment.message_id, []).append(
                    AttachmentPreview(
                        id=attachment.id,
                        filename=attachment.filename,
                        content_type=attachment.content_type,
                        status=attachment.status,
                        preview_url=preview_url
                    )
                )

        if not before:
            session_service = request.app.state.session_service
            warm_result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(desc(Message.created_at))
                .limit(settings.MAX_HISTORY_MESSAGES * 2)
            )
            warm_messages = list(reversed(warm_result.scalars().all()))
            await session_service.warm_cache(conversation_id, warm_messages)

        next_cursor = (
            messages[0].created_at.isoformat() if has_more and messages else None
        )

        return MessageHistoryResponse(
            messages=[message_to_response(m, attachments_by_message.get(m.id)) for m in messages],
            total=len(messages),
            has_more=has_more,
            next_cursor=next_cursor,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_message(
    session: AsyncSession,
    api_request: Request,
    request: ChatRequest,
):
    try:
        is_new_conversation = request.conversation_id is None

        if request.conversation_id is None:
            conversation = Conversation(
                user_id=request.user_id,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
        else:
            conversation = await get_conversation_or_404(request.conversation_id, session)
            if conversation.user_id != request.user_id:
                raise HTTPException(status_code=403, detail="Access denied")

        return EventSourceResponse(
            chat_stream_handler(
                conversation,
                request,
                session,
                api_request.app.state.assistant_graph,
                api_request.app.state.session_service,
                is_new_conversation
            ),
            media_type="text/event-stream",
            ping=10,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def send_message(
    session: AsyncSession,
    conversation_id: str,
    api_request: Request,
    request: ChatRequest,
):
    try:
        conv = await get_conversation_or_404(conversation_id, session)
        if conv.user_id != request.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        session_service = api_request.app.state.session_service
        retrieval = api_request.app.state.retrieval_service
        chat_service = api_request.app.state.chat_service
        translator: TranslatorService = api_request.app.state.translator_service

        plan = resolve_translation_plan(request.language, request.message)

        try:
            english_query = (
                (await translator.to_english(request.message)).text or request.message
                if plan.translate_inbound
                else request.message
            )
        except TranslationError:
            raise HTTPException(
                status_code=502, detail="Translation service is unavailable"
            )

        user_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=english_query,
            token_count=int(len(english_query.split()) * 1.3),
        )

        if english_query != request.message:
            user_msg.translated_content = request.message
            user_msg.is_translated = True

        session.add(user_msg)
        await session.flush()

        await link_attachments_to_message(
            session, request.attachment_ids, conversation_id, user_msg.id
        )

        try:
            await session_service.append_message(
                conversation_id, "user", english_query
            )

            chunks = await retrieval.search(query=english_query)
            history = await session_service.get_langchain_history(conversation_id)
            response = await chat_service.get_response(
                english_query, history, chunks, mode="rag"
            )

        except Exception as e:
            await session.rollback()
            logger.error(f"Chat pipeline failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to generate the response"
            )

        sources = [
            {
                "page": c.metadata.get("page_start", 0),
                "score": c.score,
            }
            for c in chunks
        ]

        translated_reply = ""
        translation_failed = False

        if plan.translate_outbound:
            try:
                translated_reply = await translator.to_sinhala(response)
            except TranslationError:
                logger.exception("Outbound translation failed; returning English")
                translation_failed = True

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=response,
            token_count=int(len(response.split()) * 1.3),
            message_metadata={"sources": sources, "rag_used": len(chunks) > 0},
        )

        if translated_reply:
            assistant_msg.translated_content = translated_reply
            assistant_msg.is_translated = True

        session.add(assistant_msg)
        conv.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(assistant_msg)

        await session_service.append_message(conversation_id, "assistant", response)

        result = message_to_response(assistant_msg)
        result.translation_failed = translation_failed

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

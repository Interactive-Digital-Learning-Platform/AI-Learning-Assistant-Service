import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import HTTPException, Request
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.conversation_model import Conversation
from app.models.message_model import Message, MessageRole
from app.schemas.conversation import (
    ChatRequest,
    ConversationCreate,
)
from app.schemas.message import MessageHistoryResponse
from app.utils.conversation import conversation_to_response, get_conversation_or_404
from app.utils.message import chat_stream_handler, message_to_response

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
            messages=[message_to_response(m) for m in messages],
            total=len(messages),
            has_more=has_more,
            next_cursor=next_cursor,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_message(
    session: AsyncSession,
    conversation_id: str,
    api_request: Request,
    request: ChatRequest,
):
    try:
        conv = await get_conversation_or_404(conversation_id, session)
        if conv.user_id != request.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return EventSourceResponse(
            chat_stream_handler(
                conv,
                request,
                session,
                api_request.app.state.assistant_graph,
                api_request.app.state.session_service,
            ),
            media_type="text/event-stream",
        )

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

        user_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=request.message,
            token_count=int(len(request.message.split()) * 1.3),
        )
        session.add(user_msg)
        await session.flush()

        try:
            await session_service.append_message(
                conversation_id, "user", request.message
            )

            chunks = await retrieval.search(query=request.message)
            history = await session_service.get_langchain_history(conversation_id)
            response = await chat_service.get_response(request.message, history, chunks)

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

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=response,
            token_count=int(len(response.split()) * 1.3),
            metadata={"sources": sources, "rag_used": len(chunks) > 0},
        )
        session.add(assistant_msg)
        conv.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(assistant_msg)

        await session_service.append_message(conversation_id, "assistant", response)

        return message_to_response(assistant_msg)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

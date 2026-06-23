import logging
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from app.schemas.conversation import (
    ConversationResponse,
    ConversationCreate,
    ChatRequest,
)
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from app.core.database import get_async_session
from app.models.conversation_model import Conversation
from app.utils.conversation import conversation_to_response, get_conversation_or_404
from app.utils.message import message_to_response, sse_generator
from app.models.message_model import Message, MessageRole
from app.services.session_service import SessionService
from app.schemas.message import MessageHistoryResponse, MessageResponse
from typing import Optional
from datetime import datetime, timezone
from sse_starlette.sse import EventSourceResponse
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Chat"])


@router.post("/", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    data: ConversationCreate,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    conversation = Conversation(
        user_id=data.user_id,
    )

    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)

    return conversation_to_response(conversation, message_count=0)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):

    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
        .offset(offset)
    )
    conversations = result.scalars().all()

    responses = []
    for conv in conversations:
        count_result = await session.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        )
        count = count_result.scalar() or 0
        responses.append(conversation_to_response(conv, message_count=count))

    return responses


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    conv = await get_conversation_or_404(conversation_id, session)
    count_result = await session.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conv.id)
    )
    count = count_result.scalar() or 0
    return conversation_to_response(conv, message_count=count)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):

    conv = await get_conversation_or_404(conversation_id, session)

    session_service = SessionService()
    await session_service.clear(conversation_id)

    await session.delete(conv)
    await session.commit()
    logger.info(f"Conversation deleted — id={conversation_id}")


@router.get("/{conversation_id}/messages", response_model=MessageHistoryResponse)
async def get_messages(
    conversation_id: str,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    limit: int = Query(50, ge=1, le=100),
    before: Optional[str] = Query(
        None, description="ISO timestamp cursor — returns messages before this time"
    ),
):
    """
    Load message history for a conversation.

    Messages are always returned in chronological order (oldest first).
    Use 'before' cursor for pagination when loading older messages.

    On conversation open — call without 'before' to get latest N messages.
    When user scrolls up — call with before=<oldest_message_created_at>
    """
    conv = await get_conversation_or_404(conversation_id, db)

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

    messages_result = await db.execute(
        query.order_by(desc(Message.created_at)).limit(limit + 1)
    )
    messages = messages_result.scalars().all()

    has_more = len(messages) > limit
    if has_more:
        messages = messages[:limit]

    messages = list(reversed(messages))

    if not before:
        session_service = SessionService()
        if not await session_service.cache_exists(conversation_id):
            await session_service.warm_cache(conversation_id, messages)

    next_cursor = messages[0].created_at.isoformat() if has_more and messages else None

    return MessageHistoryResponse(
        messages=[message_to_response(m) for m in messages],
        total=len(messages),
        has_more=has_more,
        next_cursor=next_cursor,
    )


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    api_request: Request,
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):

    conv = await get_conversation_or_404(conversation_id, session)
    if conv.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    embedder = api_request.app.state.embedder

    return EventSourceResponse(
        sse_generator(conv, request, session, embedder),
        media_type="text/event-stream",
    )


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: str,
    api_request: Request,
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):

    conv = await get_conversation_or_404(conversation_id, session)
    if conv.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    session_service = SessionService()
    embedder = api_request.app.state.embedder
    retrieval = RetrievalService(embedder=embedder)
    chat_service = ChatService()

    user_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content=request.message,
        token_count=int(len(request.message.split()) * 1.3),
    )
    session.add(user_msg)
    await session.flush()

    await session_service.append_message(conversation_id, "user", request.message)

    chunks = await retrieval.search(
        query=request.message,
        filename=conv.filename,
    )
    history = await session_service.get_langchain_history(conversation_id)
    response = await chat_service.get_response(request.message, history, chunks)

    sources = [
        {
            "filename": c.metadata.get("filename", ""),
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
    conv.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(assistant_msg)

    await session_service.append_message(conversation_id, "assistant", response)

    return message_to_response(assistant_msg)

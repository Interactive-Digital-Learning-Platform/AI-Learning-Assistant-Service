from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers import chat_controllers
from app.core.database import get_async_session
from app.schemas.conversation import (
    ChatRequest,
    ConversationCreate,
    ConversationResponse,
)
from app.schemas.message import MessageHistoryResponse, MessageResponse

router = APIRouter(prefix="/conversations", tags=["Chat"])


@router.post("/", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    data: ConversationCreate,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await chat_controllers.create_conversation(session, data)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await chat_controllers.list_conversations(session, user_id, limit, offset)


@router.get("/messages/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await chat_controllers.get_message(session, message_id, request)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await chat_controllers.get_conversation(session, conversation_id)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await chat_controllers.delete_conversation(session, request, conversation_id)


@router.get("/{conversation_id}/messages", response_model=MessageHistoryResponse)
async def get_messages(
    conversation_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    limit: int = Query(50, ge=1, le=100),
    before: Optional[str] = Query(
        None, description="ISO timestamp cursor — returns messages before this time"
    ),
):

    return await chat_controllers.get_messages(
        session, conversation_id, request, limit, before
    )


@router.post("/messages/stream")
async def stream_message(
    api_request: Request,
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):

    return await chat_controllers.stream_message(
        session, api_request, request
    )


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: str,
    api_request: Request,
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):

    return await chat_controllers.send_message(
        session, conversation_id, api_request, request
    )

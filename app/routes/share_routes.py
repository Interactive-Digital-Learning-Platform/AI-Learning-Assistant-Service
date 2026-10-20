from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers import share_controllers
from app.core.database import get_async_session
from app.schemas.share import (
    SharedChatResponse,
    ShareLinkRequest,
    ShareLinkResponse,
)

router = APIRouter(tags=["Share"])


@router.post(
    "/conversations/{conversation_id}/share", response_model=ShareLinkResponse
)
async def create_share_link(
    conversation_id: str,
    data: ShareLinkRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await share_controllers.create_share_link(
        session, conversation_id, data.user_id
    )


@router.delete("/conversations/{conversation_id}/share", status_code=204)
async def revoke_share_link(
    conversation_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: str = Query(...),
):
    return await share_controllers.revoke_share_link(
        session, conversation_id, user_id
    )


# Public: anyone holding the token can read the snapshot.
@router.get("/shared/{token}", response_model=SharedChatResponse)
async def get_shared_chat(
    token: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    return await share_controllers.get_shared_chat(session, token)

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.controllers import attachment_controllers
from app.core.database import get_async_session
from app.schemas.attachment import AttachmentResponse

router = APIRouter(prefix="/conversations/{conversation_id}/attachments", tags=["Attachments"])


@router.post("/", response_model=AttachmentResponse)
async def upload_attachment(
    conversation_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    return await attachment_controllers.upload_attachment(
        session, request, conversation_id, user_id, file
    )


@router.get("/{attachment_id}", response_model=AttachmentResponse)
async def get_attachment(
    attachment_id: str,
    request: Request,
    conversation_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: str = Query(...)
):
    return await attachment_controllers.get_attachment(
        session, request, conversation_id, attachment_id, user_id
    )
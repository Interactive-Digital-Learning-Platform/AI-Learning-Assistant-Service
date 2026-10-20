import logging
import secrets
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attachment_model import Attachment
from app.models.message_model import Message, MessageRole
from app.models.shared_chat_model import SharedChat
from app.schemas.share import (
    SharedAttachment,
    SharedChatResponse,
    SharedMessage,
    ShareLinkResponse,
)
from app.utils.conversation import get_conversation_or_404
from app.utils.message import message_to_response

logger = logging.getLogger(__name__)

# ~128 bits of randomness, URL-safe — the token is the only thing guarding a link.
SHARE_TOKEN_BYTES = 16


async def create_share_link(
    session: AsyncSession, conversation_id: str, user_id: str
) -> ShareLinkResponse:
    try:
        conv = await get_conversation_or_404(conversation_id, session)
        if conv.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        snapshot = await _snapshot_messages(session, conv.id)
        if not snapshot:
            raise HTTPException(
                status_code=400, detail="This conversation has no messages to share"
            )

        existing = await _get_share_for_conversation(session, conv.id)
        if existing:
            # Re-sharing refreshes the snapshot but keeps the same link.
            existing.messages = snapshot
            existing.title = conv.title
            existing.updated_at = datetime.now(UTC)
            shared = existing
        else:
            shared = SharedChat(
                token=secrets.token_urlsafe(SHARE_TOKEN_BYTES),
                conversation_id=conv.id,
                user_id=user_id,
                title=conv.title,
                messages=snapshot,
            )
            session.add(shared)

        try:
            await session.commit()
        except IntegrityError:
            # A concurrent request created this conversation's link first; it
            # holds an equivalent snapshot, so hand back that link.
            await session.rollback()
            shared = await _get_share_for_conversation(session, conv.id)
            if shared is None:
                raise

        logger.info(f"Share link saved — conversation={conversation_id}")
        return ShareLinkResponse(
            token=shared.token,
            shared_at=shared.updated_at,
            message_count=len(shared.messages),
        )

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.exception(f"Share link creation failed — conversation={conversation_id}")
        raise HTTPException(status_code=500, detail=str(e))


async def revoke_share_link(
    session: AsyncSession, conversation_id: str, user_id: str
) -> None:
    try:
        conv = await get_conversation_or_404(conversation_id, session)
        if conv.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        await session.execute(
            delete(SharedChat).where(SharedChat.conversation_id == conv.id)
        )
        await session.commit()
        logger.info(f"Share link revoked — conversation={conversation_id}")

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.exception(f"Share link revoke failed — conversation={conversation_id}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_shared_chat(session: AsyncSession, token: str) -> SharedChatResponse:
    try:
        result = await session.execute(
            select(SharedChat).where(SharedChat.token == token)
        )
        shared = result.scalar_one_or_none()
        if shared is None:
            raise HTTPException(status_code=404, detail="Shared chat not found")

        return SharedChatResponse(
            title=shared.title,
            shared_at=shared.updated_at,
            messages=[SharedMessage.model_validate(m) for m in shared.messages],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Shared chat lookup failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_share_for_conversation(
    session: AsyncSession, conversation_uuid
) -> SharedChat | None:
    result = await session.execute(
        select(SharedChat).where(SharedChat.conversation_id == conversation_uuid)
    )
    return result.scalar_one_or_none()


async def _snapshot_messages(session: AsyncSession, conversation_uuid) -> list[dict]:
    messages_result = await session.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_uuid,
            Message.role != MessageRole.SYSTEM,
        )
        .order_by(Message.created_at)
    )
    messages = messages_result.scalars().all()

    attachment_result = await session.execute(
        select(Attachment).where(
            Attachment.conversation_id == conversation_uuid,
            Attachment.message_id.is_not(None),
        )
    )
    attachments_by_message: dict = {}
    for attachment in attachment_result.scalars().all():
        attachments_by_message.setdefault(attachment.message_id, []).append(
            SharedAttachment(
                filename=attachment.filename,
                content_type=attachment.content_type,
            )
        )

    snapshot: list[dict] = []
    for message in messages:
        if not message.content.strip():
            continue
        shared_message = SharedMessage(
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            sources=message_to_response(message).sources,
            attachments=attachments_by_message.get(message.id, []),
            is_translated=bool(message.is_translated),
            translated_content=message.translated_content,
        )
        snapshot.append(shared_message.model_dump(mode="json"))

    return snapshot

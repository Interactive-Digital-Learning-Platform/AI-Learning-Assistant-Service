from app.models.conversation_model import Conversation
from app.schemas.conversation import ConversationResponse
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
import uuid

def conversation_to_response(
    conv: Conversation,
    message_count: int,
) -> ConversationResponse:
    return ConversationResponse(
        conversation_id = conv.id,
        user_id         = conv.user_id,
        title           = conv.title,
        created_at      = conv.created_at,
        updated_at      = conv.updated_at,
        message_count   = message_count,
    )

async def get_conversation_or_404(
    conversation_id: str,
    db: AsyncSession,
) -> Conversation:
    try:
        cid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")
 
    conv = await db.get(Conversation, cid)
    if not conv:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    return conv
 
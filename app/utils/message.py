import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.message_model import Message, MessageRole
from app.schemas.conversation import ChatRequest
from app.schemas.message import MessageResponse, SourceCitation
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)


def message_to_response(msg: Message) -> MessageResponse:
    meta = msg.message_metadata or {}
    sources = [
        SourceCitation(
            filename=s.get("filename", ""),
            page=s.get("page", 0),
            score=s.get("score", 0.0),
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
    )


async def sse_generator(
    conv,
    request: ChatRequest,
    session: AsyncSession,
    session_service: SessionService,
    retrieval_service: RetrievalService,
    chat_service: ChatService,
):

    full_response = ""
    assistant_msg = None
    conversation_id = str(conv.id)
    _committed = False

    try:
        user_message = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=request.message,
            token_count=int(len(request.message.split()) * 1.3),
        )
        session.add(user_message)
        await session.flush()

        try:
            chunks = await asyncio.wait_for(
                retrieval_service.search(query=request.message),
                timeout=10,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Retrieval timed out — conversation={conversation_id}",
            )
            chunks = []
        except Exception as exc:
            logger.warning(
                f"Retrieval failed — conversation={conversation_id}: {exc}",
                exc_info=True,
            )
            chunks = []

        history = await session_service.get_langchain_history(conversation_id)

        deadline = time.monotonic() + 60.0
        
        try:
            async for token in chat_service.stream_response(
                message=request.message,
                history=history,
                context=chunks,
            ):
                if time.monotonic() > deadline:
                    logger.error(f"LLM stream timeout — conversation={conversation_id}")
                    raise asyncio.TimeoutError("LLM response exceeded time limit")
                    
                full_response += token
                yield {
                    "data": json.dumps(
                        {
                            "type": "token",
                            "token": token,
                        }
                    )
                }
        except Exception as exc:
            logger.error(
                f"LLM stream failed — conversation={conversation_id}: {exc}",
                exc_info=True,
            )
            await session.rollback()
            _committed = True
            yield {
                "data": json.dumps(
                    {
                        "type": "error",
                        "error": "AI response timed out. Please try again.",
                    }
                )
            }
            return

        sources = [
            {
                "filename": c.metadata.get("filename", ""),
                "page": c.metadata.get("page_start", 0),
                "score": c.score,
            }
            for c in chunks
        ]

        if not conv.title:
            conv.title = (
                request.message[:97] + "..."
                if len(request.message) > 100
                else request.message
            )

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            token_count=int(len(full_response.split()) * 1.3),
            metadata={
                "sources": sources,
                "model": settings.GROQ_MODEL,
                "rag_used": len(chunks) > 0,
                "chunks_used": len(chunks),
            },
        )

        session.add(assistant_msg)
        conv.updated_at = datetime.now(timezone.utc)

        await session.commit()
        _committed = True
        await session.refresh(assistant_msg)

        await session_service.append_message(conversation_id, "user", request.message)
        await session_service.append_message(
            conversation_id, "assistant", full_response
        )

        yield {
            "data": json.dumps(
                {
                    "type": "done",
                    "message_id": str(assistant_msg.id),
                    "sources": sources,
                }
            )
        }

        logger.info(
            f"Message complete — conversation={conversation_id} "
            f"rag={'yes' if chunks else 'no'} "
            f"chunks={len(chunks)} tokens≈{len(full_response.split())}"
        )

    except Exception as exc:
        logger.error(
            f"Stream error — conversation={conversation_id}: {exc}",
            exc_info=True,
        )

        if not _committed:
            await session.rollback()
            
        yield {
            "data": json.dumps(
                {
                    "type": "error",
                    "error": "An error occurred while generating the response.",
                }
            )
        }

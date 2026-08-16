import asyncio
import json
import logging
from datetime import UTC, datetime

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_model import Conversation
from app.models.message_model import Message, MessageRole
from app.schemas.agent_state import AgentState
from app.schemas.conversation import ChatRequest
from app.schemas.message import MessageResponse, SourceCitation
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
    committed = False

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

        initial_state: AgentState = {
            "conversation_id": conversation_id,
            "user_id": str(conv.user_id),
            "user_message": request.message,
            "history": [],
            "intent": "general",
            "retrieved_chunks": [],
            "sources": [],
            "rewritten_query": "",
            "context": [],
            "response": "",
            "rag_used": False,
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
                        yield {
                            "data": json.dumps({
                                "type": "token",
                                "token": token,
                            })
                        }

        final_state = await stream.output()

        if final_state is None:
            raise RuntimeError("Graph finished without producing a final state")


        sources = final_state.get("sources", [])

        if not conv.title:
            conv.title = (
                request.message[:97] + "..."
                if len(request.message) > 100
                else request.message
            )

        
        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=final_state.get("response", full_response),
            token_count=int(len(full_response.split()) * 1.3),
            metadata={
                "sources": sources,
                "intent": final_state.get("intent"),
                "rag_used": final_state.get("rag_used", False),
                "rewritten_query": final_state.get("rewritten_query", ""),
            },
        )

        session.add(assistant_msg)
        conv.updated_at = datetime.now(UTC)

        await session.commit()
        committed = True
        await session.refresh(assistant_msg)

        try:
            await session_service.append_message(
                conversation_id,
                MessageRole.USER,
                request.message
            )

            await session_service.append_message(
                conversation_id,
                MessageRole.ASSISTANT,
                full_response
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

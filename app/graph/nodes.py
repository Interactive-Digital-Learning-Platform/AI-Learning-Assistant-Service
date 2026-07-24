import logging
from typing import Any, Literal

from langgraph.types import Command

from app.schemas.agent_state import AgentState
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)


class GraphNodes:
    def __init__(
        self,
        session_service: SessionService,
        retrieval_service: RetrievalService,
        chat_service: ChatService,
    ):
        self.session_service = session_service
        self.retrieval_service = retrieval_service
        self.chat_service = chat_service

    async def load_memory_node(
        self, state: AgentState
    ) -> Command[Literal["classify_intent"]]:
        history = await self.session_service.get_langchain_history(
            state["conversation_id"]
        )

        return Command(update={"history": history}, goto="classify_intent")

    async def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        rewritten_query = await self.chat_service.rewrite_query(
            user_query=state["user_message"], history=state["history"]
        )

        logger.info(
            f"Query rewritten: '{state['user_message']}' -> '{rewritten_query}'"
        )

        return {"rewritten_query": rewritten_query}

    async def retrieve_docs_node(self, state: AgentState) -> dict[str, Any]:
        query = state["rewritten_query"] or state["user_message"]
        chunks = await self.retrieval_service.search(query=query)
        context = self.chat_service.format_context(chunks=chunks)
        sources = [
            {
                "filename": c.metadata.get("filename", ""),
                "page": c.metadata.get("page_start", 0),
                "score": c.score,
            }
            for c in chunks
        ]

        return {
            "sources": sources,
            "context": context,
            "chunks_retrieved": chunks,
            "rag_used": len(chunks) > 0,
        }

    async def generate_response_node(self, state: AgentState) -> dict[str, Any]:
        response = await self.chat_service.get_response(
            message=state["rewritten_query"] or state["user_message"],
            history=state["history"],
            context=state["context"],
        )

        return {"response": response}

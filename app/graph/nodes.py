import logging
from asyncio import wait_for
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
        logger.info("Load memory node is reached")
        history = await self.session_service.get_langchain_history(
            state["conversation_id"]
        )

        return Command(update={"history": history}, goto="classify_intent")

    async def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        response = await self.chat_service.rewrite_query(
            user_query=state["user_message"], history=state["history"]
        )

        logger.info(
            f"Query rewritten: '{state['user_message']}' -> '{response.rewritten_query}'"
        )

        return {"rewritten_query": response.rewritten_query}

    async def retrieve_docs_node(self, state: AgentState) -> dict[str, Any]:
        query = state["rewritten_query"] or state["user_message"]

        try:
            chunks = await wait_for(
                self.retrieval_service.search(query=query),
                timeout=15.0
            )

        except Exception:
            logger.exception(
                "Document retrieval failed; continuing without context"
            )
            chunks = []
            
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
            "context": chunks,
            "retrieved_chunks": chunks,
            "rag_used": True if state["intent"] == "rag" else False,
        }

    async def generate_response_node(self, state: AgentState) -> dict[str, Any]:
        logger.info("Response node is reached")
        response = await self.chat_service.get_response(
            message=state["rewritten_query"] or state["user_message"],
            history=state["history"],
            context=state["context"],
            mode=state["intent"]
        )

        return {"response": response}

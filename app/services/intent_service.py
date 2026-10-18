import logging
from asyncio import wait_for
from typing import Literal

from langgraph.types import Command

from app.core.config import settings
from app.prompts.graph_prompts import attachments_note, build_query_analysis_prompt
from app.schemas.agent_state import AgentState, QueryAnalysis

logger = logging.getLogger(__name__)

AnalysisRoute = Literal["generate_response", "retrieve_docs", "web_search"]

_GOTO_BY_INTENT: dict[str, AnalysisRoute] = {
    "general": "generate_response",
    "rag": "retrieve_docs",
    "web_search": "web_search",
    "generate_pdf": "retrieve_docs",
}

_MAX_SUB_QUERIES = 3


class IntentService:
    def __init__(
        self,
        llm,
        web_search_enabled: bool = False,
        pdf_generation_enabled: bool = False,
        timeout_seconds: float = settings.QUERY_ANALYSIS_TIMEOUT_SECONDS,
    ):
        self.llm = llm
        self.web_search_enabled = web_search_enabled
        self.pdf_generation_enabled = pdf_generation_enabled
        self.timeout_seconds = timeout_seconds
        self._prompt = build_query_analysis_prompt(
            web_enabled=web_search_enabled,
            pdf_enabled=pdf_generation_enabled,
        )

    async def classify_intent(
        self, state: AgentState
    ) -> Command[AnalysisRoute]:
        message = state["user_message"]
        history = state["history"]
        forced_rag = bool(state.get("message_has_attachments"))

        if forced_rag and not history:
            analysis = QueryAnalysis(intent="rag")
        else:
            analysis = await self._analyse(state)

        intent = "rag" if forced_rag else analysis.intent

        if intent == "web_search" and not self.web_search_enabled:
            intent = "rag"

        if intent == "generate_pdf" and not self.pdf_generation_enabled:
            intent = "general"

        standalone = (
            analysis.standalone_query.strip()
            if history or intent == "generate_pdf"
            else ""
        )
        query = standalone or message
        sub_queries = (
            self._clean_sub_queries(analysis.sub_queries, query)
            if intent in ("rag", "generate_pdf")
            else []
        )

        subject = analysis.subject if intent in ("rag", "generate_pdf") else None

        logger.info(
            f"Query analysed — intent={intent} forced_rag={forced_rag} "
            f"query='{query[:80]}' sub_queries={sub_queries} subject={subject}"
        )

        return Command(
            update={
                "intent": intent,
                "rewritten_query": query,
                "sub_queries": sub_queries,
                "subject": subject,
            },
            goto=_GOTO_BY_INTENT[intent],
        )

    async def _analyse(self, state: AgentState) -> QueryAnalysis:
        structured_llm = self.llm.with_structured_output(QueryAnalysis, method="json_mode")
        chain = self._prompt | structured_llm

        try:
            return await wait_for(
                chain.ainvoke(
                    {
                        "user_message": state["user_message"],
                        "history": state["history"],
                        "attachments_note": attachments_note(state.get("attachment_names", {})),
                    }
                ),
                timeout=self.timeout_seconds,
            )
        except Exception:
            logger.exception("Query analysis failed; defaulting to rag with the original message")
            return QueryAnalysis(intent="rag")

    @staticmethod
    def _clean_sub_queries(sub_queries: list[str], query: str) -> list[str]:
        cleaned: list[str] = []
        seen = {query.strip().lower()}

        for sub_query in sub_queries:
            text = str(sub_query).strip()
            key = text.lower()

            if not text or key in seen:
                continue

            seen.add(key)
            cleaned.append(text)

        return cleaned[:_MAX_SUB_QUERIES] if len(cleaned) >= 2 else []

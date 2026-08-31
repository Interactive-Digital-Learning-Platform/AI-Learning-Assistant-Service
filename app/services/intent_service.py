import logging
from typing import Literal

from langgraph.types import Command

from app.prompts.graph_prompts import build_classification_prompt
from app.schemas.agent_state import AgentState, QueryClassification

logger = logging.getLogger(__name__)

_GOTO_BY_INTENT = {
    "general": "generate_response",
    "rag": "rewrite_query",
    "web_search": "rewrite_query",
    "generate_pdf": "generate_document",
}


class IntentService:
    def __init__(
        self,
        llm,
        web_search_enabled: bool = False,
        pdf_generation_enabled: bool = False,
    ):
        self.llm = llm
        self.web_search_enabled = web_search_enabled
        self.pdf_generation_enabled = pdf_generation_enabled
        self._prompt = build_classification_prompt(
            web_enabled=web_search_enabled,
            pdf_enabled=pdf_generation_enabled,
        )

    async def classify_intent(
        self, state: AgentState
    ) -> Command[Literal["generate_response", "rewrite_query", "generate_document"]]:

        if state.get("has_attachments"):
            return Command(
                update={
                    "intent": "rag"
                },
                goto="rewrite_query"
            )

        structured_llm = self.llm.with_structured_output(QueryClassification, method="json_mode")

        chain = self._prompt | structured_llm

        classification = await chain.ainvoke(
            {"user_message": state["user_message"], "history": state["history"]}
        )

        intent = classification.intent

        if intent == "web_search" and not self.web_search_enabled:
            intent = "rag"

        if intent == "generate_pdf" and not self.pdf_generation_enabled:
            intent = "general"

        goto = _GOTO_BY_INTENT.get(intent)

        if goto is None:
            raise ValueError(f"Unknown intent: {classification.intent}")

        return Command(update={"intent": intent}, goto=goto)

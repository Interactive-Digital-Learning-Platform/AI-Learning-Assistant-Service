import logging
from typing import Literal

from langgraph.types import Command

from app.prompts.graph_prompts import (
    classification_prompt,
    classification_prompt_with_web,
)
from app.schemas.agent_state import AgentState, QueryClassification

logger = logging.getLogger(__name__)

_GOTO_BY_INTENT = {
    "general": "generate_response",
    "rag": "rewrite_query",
    "web_search": "rewrite_query",
}


class IntentService:
    def __init__(self, llm, web_search_enabled: bool = False):
        self.llm = llm
        self.web_search_enabled = web_search_enabled

    async def classify_intent(
        self, state: AgentState
    ) -> Command[Literal["generate_response", "rewrite_query"]]:

        if state.get("has_attachments"):
            return Command(
                update={
                    "intent": "rag"
                },
                goto="rewrite_query"
            )

        prompt = (
            classification_prompt_with_web
            if self.web_search_enabled
            else classification_prompt
        )

        structured_llm = self.llm.with_structured_output(QueryClassification, method="json_mode")

        chain = prompt | structured_llm

        classification = await chain.ainvoke(
            {"user_message": state["user_message"], "history": state["history"]}
        )

        intent = classification.intent

        if intent == "web_search" and not self.web_search_enabled:
            intent = "rag"

        goto = _GOTO_BY_INTENT.get(intent)

        if goto is None:
            raise ValueError(f"Unknown intent: {classification.intent}")

        return Command(update={"intent": intent}, goto=goto)

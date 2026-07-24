from typing import Literal

from langgraph.types import Command

from app.prompts.graph_prompts import classification_prompt
from app.schemas.agent_state import AgentState, QueryClassification


class IntentService:
    def __init__(self, llm):
        self.llm = llm

    async def classify_intent(
        self, state: AgentState
    ) -> Command[Literal["generate_response", "rewrite_query"]]:

        structured_llm = self.llm.with_structured_output(QueryClassification)

        chain = classification_prompt | structured_llm

        classification = await chain.ainvoke(
            {"user_message": state["user_message"], "history": state["history"]}
        )

        if classification["intent"] == "general":
            goto = "generate_response"

        elif classification["intent"] == "rag":
            goto = "rewrite_query"

        else:
            raise ValueError(f"Unknown intent: {classification['intent']}")

        return Command(update={"intent": classification["intent"]}, goto=goto)

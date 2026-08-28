from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.graph.nodes import GraphNodes
from app.schemas.agent_state import AgentState
from app.services.intent_service import IntentService


def create_assistant_graph(
    nodes: GraphNodes, intent_service: IntentService
) -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", nodes.load_memory_node)
    workflow.add_node("check_attachments", nodes.check_attachments_node)
    workflow.add_node("classify_intent", intent_service.classify_intent)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node(
        "retrieve_docs",
        nodes.retrieve_docs_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node("generate_response", nodes.generate_response_node)

    workflow.add_edge(START, "load_memory")
    workflow.add_edge("rewrite_query", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "generate_response")
    workflow.add_edge("generate_response", END)

    assistant_graph = workflow.compile()

    return assistant_graph

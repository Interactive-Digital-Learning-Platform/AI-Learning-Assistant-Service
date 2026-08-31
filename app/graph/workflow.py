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

    workflow.add_node("translate_input", nodes.ai_translator_node)
    workflow.add_node("translate_output", nodes.ai_translator_node)
    workflow.add_node("load_memory", nodes.load_memory_node)
    workflow.add_node("check_attachments", nodes.check_attachments_node)
    workflow.add_node("classify_intent", intent_service.classify_intent)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node(
        "retrieve_docs",
        nodes.retrieve_docs_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node(
        "web_search",
        nodes.web_search_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node(
        "generate_document",
        nodes.generate_document_node,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    workflow.add_node("generate_response", nodes.generate_response_node)

    workflow.add_edge(START, "translate_input")
    workflow.add_edge("translate_input", "load_memory")
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_document", "generate_response")
    workflow.add_edge("generate_response", "translate_output")
    workflow.add_edge("translate_output", END)

    assistant_graph = workflow.compile()

    return assistant_graph

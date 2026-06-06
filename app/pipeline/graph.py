from langgraph.graph import StateGraph, END
from app.pipeline.state import RAGState
from app.pipeline import nodes


def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    workflow.add_node("query_rewriter", nodes.query_rewriter)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("reranker", nodes.reranker_node)
    workflow.add_node("answer_generation", nodes.answer_generation)
    workflow.add_node("no_context_fallback", nodes.no_context_fallback)
    workflow.add_node("citation_formatter", nodes.citation_formatter)

    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("retriever", "reranker")

    workflow.add_conditional_edges(
        "reranker",
        nodes.should_generate,
        {
            "generate": "answer_generation",
            "no_context_fallback": "no_context_fallback",
        },
    )

    workflow.add_edge("answer_generation", "citation_formatter")
    workflow.add_edge("no_context_fallback", END)
    workflow.add_edge("citation_formatter", END)

    return workflow.compile()


rag_graph = build_rag_graph()

from langgraph.graph import StateGraph, END
from app.pipeline.state import RAGState
from app.pipeline import nodes


def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    # Agents
    workflow.add_node("guardrails_agent", nodes.guardrails_agent)
    workflow.add_node("reject_query", nodes.reject_query)
    workflow.add_node("query_rewriter", nodes.query_rewriter)
    workflow.add_node("retrieval_strategist", nodes.retrieval_strategist)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("reranker", nodes.reranker_node)
    workflow.add_node("answer_generation", nodes.answer_generation)
    workflow.add_node("no_context_fallback", nodes.no_context_fallback)
    workflow.add_node("quality_reviewer_agent", nodes.quality_reviewer_agent)
    workflow.add_node("citation_formatter", nodes.citation_formatter)

    # Entry → Guardrails
    workflow.set_entry_point("guardrails_agent")

    # Guardrails conditional
    workflow.add_conditional_edges(
        "guardrails_agent",
        nodes.guardrails_router,
        {
            "rewrite": "query_rewriter",
            "reject_query": "reject_query",
        },
    )

    # Reject → END (skip pipeline)
    workflow.add_edge("reject_query", END)

    # Rewrite → Strategist
    workflow.add_edge("query_rewriter", "retrieval_strategist")

    # Strategist → Retriever (passes doc_id if doc-specific)
    workflow.add_edge("retrieval_strategist", "retriever")

    # Retriever → Reranker
    workflow.add_edge("retriever", "reranker")

    # Reranker → conditional to answer or fallback
    workflow.add_conditional_edges(
        "reranker",
        nodes.should_generate,
        {
            "generate": "answer_generation",
            "no_context_fallback": "no_context_fallback",
        },
    )

    # Fallback → Quality (still review fallback answer) → Format
    workflow.add_edge("no_context_fallback", "quality_reviewer_agent")

    # Answer gen → Quality Reviewer
    workflow.add_edge("answer_generation", "quality_reviewer_agent")

    # Quality reviewer → conditional: format or regenerate
    workflow.add_conditional_edges(
        "quality_reviewer_agent",
        nodes.quality_router,
        {
            "format": "citation_formatter",
            "regenerate": "answer_generation",
        },
    )

    workflow.add_edge("citation_formatter", END)

    return workflow.compile()


rag_graph = build_rag_graph()

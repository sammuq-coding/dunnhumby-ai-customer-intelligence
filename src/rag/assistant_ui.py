"""Chat interface used as an additive page in the existing dashboard."""
from __future__ import annotations

import streamlit as st

from src.rag.assistant import answer_question, provider_from_environment
from src.rag.context_builder import build_knowledge_base


def render_assistant_page(data: dict) -> None:
    st.title("AI Analytics Assistant")
    st.caption("Ask about the saved customer, campaign, model, calibration, and finance artifacts.")
    provider = provider_from_environment()
    if provider is None:
        st.info("No LLM endpoint is configured. Deterministic analytics tools and documentation retrieval are active; no external API is called.")
    else:
        st.info("Configured LLM is used only to route questions and explain retrieved evidence. Quantitative answers come from deterministic project tools.")

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = []
    if "rag_knowledge_base" not in st.session_state:
        st.session_state["rag_knowledge_base"] = build_knowledge_base(data)
    for message in st.session_state["rag_chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source"):
                st.caption("Source: " + ", ".join(message["source"]))
                with st.expander("Evidence details"):
                    st.caption(f"Evidence type: {message.get('evidence_type', 'Retrieved documentation')} · Route: {message.get('tool') or 'knowledge retrieval'} · Mode: {message.get('mode')}")
                    if message.get("evidence") is not None:
                        st.json(message["evidence"])
                    for doc in message.get("retrieved_documents", []):
                        st.caption(f"{doc['source']} · {doc['title']} · retrieval score {doc['score']:.2f}")

    question = st.chat_input("Ask a question about the project analytics")
    if question:
        st.session_state["rag_chat_history"].append({"role": "user", "content": question})
        try:
            result = answer_question(question, data, st.session_state["rag_knowledge_base"], provider)
        except Exception as exc:
            result = {"answer": f"I couldn't complete that request from the available artifacts: {exc}",
                      "source": [], "mode": "error", "tool": None}
        st.session_state["rag_chat_history"].append({
            "role": "assistant", "content": result["answer"], "source": result.get("source", []),
            "tool": result.get("tool"), "mode": result.get("mode"),
            "evidence_type": result.get("evidence_type"), "evidence": result.get("evidence"),
            "retrieved_documents": result.get("retrieved_documents", []),
        })
        st.rerun()

    with st.expander("Example questions"):
        st.markdown("""
        - Which five customers have the highest modeled 12-month CLV?
        - What is household 123's recent purchasing behavior?
        - Which campaigns had the highest descriptive redemption rates?
        - What is the predicted response probability for household 123?
        - How is CLV calculated?
        - How was the response model calibrated and evaluated?
        - Did campaign 18 cause sales to increase?
        """)

import os
from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from retriever import search
from state import RAGState

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Given the conversation history below, rewrite the follow-up question as a "
    "fully self-contained question that can be understood without the history. "
    "If the question is already self-contained, return it unchanged.\n\n"
    "History:\n{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)

_HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Write a short passage (2-3 sentences) that would directly answer the question "
    "if it appeared in a book. Do not explain or qualify — just write the passage.\n\n"
    "Question: {question}"
)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Conversation history:\n{history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in history)


def _rewrite(state: RAGState) -> RAGState:
    history = state.get("history") or []
    if not history:
        return {"standalone_question": state["question"]}
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    chain = _REWRITE_PROMPT | llm | StrOutputParser()
    standalone = chain.invoke({
        "history": _format_history(history),
        "question": state["question"],
    })
    return {"standalone_question": standalone}


def _hypothesize(state: RAGState) -> RAGState:
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    chain = _HYDE_PROMPT | llm | StrOutputParser()
    hypothesis = chain.invoke({"question": state["standalone_question"]})
    return {"hypothesis": hypothesis}


def _retrieve(state: RAGState) -> RAGState:
    # Search with both hypothesis and standalone question, deduplicate by content.
    # HyDE improves recall when accurate; the standalone question is a safety net.
    by_hypothesis = search(state["hypothesis"], k=8)
    by_question = search(state["standalone_question"], k=5)
    seen, merged = set(), []
    for doc in by_hypothesis + by_question:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return {"documents": merged}


def _generate(state: RAGState) -> RAGState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    chain = _ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "history": _format_history(state.get("history") or []),
        "context": context,
        "question": state["question"],
    })
    return {"answer": answer}


def build_graph():
    # TODO: pass checkpointer=MemorySaver() or PostgresSaver() here for
    #       persistent cross-session memory without changing any node logic.
    g = StateGraph(RAGState)
    g.add_node("rewrite", _rewrite)
    g.add_node("hypothesize", _hypothesize)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "hypothesize")
    g.add_edge("hypothesize", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


type StreamEvent = tuple[str, list[Document]] | tuple[str, str]


def stream_answer(
    question: str,
    history: list[tuple[str, str]],
    compiled_graph,
) -> Generator[StreamEvent, None, None]:
    """Yield typed events from a single graph run:
      ("standalone", str)       — rewritten self-contained question
      ("hypothesis", str)       — HyDE hypothetical answer used for retrieval
      ("docs", list[Document])  — retrieved chunks
      ("token", str)            — one LLM output token at a time
    """
    for mode, data in compiled_graph.stream(
        {"question": question, "history": history},
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates":
            if "rewrite" in data:
                yield ("standalone", data["rewrite"]["standalone_question"])
            elif "hypothesize" in data:
                yield ("hypothesis", data["hypothesize"]["hypothesis"])
            elif "retrieve" in data:
                yield ("docs", data["retrieve"]["documents"])
        elif mode == "messages":
            chunk, metadata = data
            if (
                metadata.get("langgraph_node") == "generate"
                and hasattr(chunk, "content")
                and chunk.content
            ):
                yield ("token", chunk.content)

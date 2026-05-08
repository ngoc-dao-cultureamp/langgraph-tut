import os
from collections.abc import Generator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from retriever import search
from state import RAGState

LLM_HOST = os.environ.get("LLM_HOST")
LLM_MODEL = os.environ.get("LLM_MODEL")

_FILTER_PROMPT = ChatPromptTemplate.from_template(
    "Is the following question related to Sherlock Holmes stories, characters, or plots? "
    "Answer with a single word: yes or no.\n\nQuestion: {question}"
)

_OFF_TOPIC_ANSWER = "I can only answer questions about Sherlock Holmes stories."

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


def _filter(state: RAGState) -> RAGState:
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_HOST, api_key="sk-local")
    chain = _FILTER_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"question": state["question"]}).strip().lower()
    if not verdict.startswith("yes"):
        return {"answer": _OFF_TOPIC_ANSWER}
    # Clear stale answer from MemorySaver so _is_off_topic routes correctly.
    return {"answer": ""}


def _is_off_topic(state: RAGState) -> str:
    return "off_topic" if state.get("answer") else "on_topic"


def _rewrite(state: RAGState) -> RAGState:
    history = state.get("history", "(none)") or "(none)"
    if history == "(none)":
        return {"standalone_question": state["question"]}
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_HOST, api_key="sk-local")
    chain = _REWRITE_PROMPT | llm | StrOutputParser()
    standalone = chain.invoke({"history": history, "question": state["question"]})
    return {"standalone_question": standalone}


def _hypothesize(state: RAGState) -> RAGState:
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_HOST, api_key="sk-local")
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
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_HOST, api_key="sk-local")
    chain = _ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "history": state.get("history", "(none)") or "(none)",
        "context": context,
        "question": state["question"],
    })
    return {"answer": answer}


def build_graph():
    # MemorySaver persists state in RAM across turns within a thread.
    # To persist across app restarts, swap for PostgresSaver — one line change.
    g = StateGraph(RAGState)
    g.add_node("filter", _filter)
    g.add_node("rewrite", _rewrite)
    g.add_node("hypothesize", _hypothesize)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_edge(START, "filter")
    g.add_conditional_edges("filter", _is_off_topic, {"off_topic": END, "on_topic": "rewrite"})
    g.add_edge("rewrite", "hypothesize")
    g.add_edge("hypothesize", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())


type StreamEvent = tuple[str, list[Document]] | tuple[str, str]


def _build_history_str(compiled_graph, config: dict) -> str:
    """Read prior (question, answer) pairs from MemorySaver for this thread."""
    try:
        states = list(compiled_graph.get_state_history(config))
        # Each state snapshot is one completed turn; collect the most recent ones
        turns = []
        for snapshot in reversed(states):
            q = snapshot.values.get("question", "")
            a = snapshot.values.get("answer", "")
            if q and a:
                turns.append(f"Q: {q}\nA: {a}")
        return "\n".join(turns) if turns else "(none)"
    except Exception:
        return "(none)"


def stream_answer(
    question: str,
    compiled_graph,
    thread_id: str,
) -> Generator[StreamEvent, None, None]:
    """Yield typed events from a single graph run:
      ("standalone", str)       — rewritten self-contained question
      ("hypothesis", str)       — HyDE hypothetical answer used for retrieval
      ("docs", list[Document])  — retrieved chunks
      ("token", str)            — one LLM output token at a time
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = _build_history_str(compiled_graph, config)

    for mode, data in compiled_graph.stream(
        {"question": question, "history": history},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates":
            if "filter" in data and data["filter"].get("answer"):
                yield ("token", data["filter"]["answer"])
            elif "rewrite" in data:
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

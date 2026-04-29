import os
from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from rag.retriever import search
from rag.state import RAGState

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _retrieve(state: RAGState) -> RAGState:
    return {"documents": search(state["question"], k=5)}


def _generate(state: RAGState) -> RAGState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    chain = _PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


type StreamEvent = tuple[str, list[Document]] | tuple[str, str]


def stream_answer(question: str, compiled_graph) -> Generator[StreamEvent, None, None]:
    """Yield ("docs", documents) once after retrieval, then ("token", text) per token.

    Uses LangGraph's dual stream mode so both node state updates and LLM tokens
    come from the same graph run without running retrieval twice.
    """
    for mode, data in compiled_graph.stream(
        {"question": question},
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates" and "retrieve" in data:
            yield ("docs", data["retrieve"]["documents"])
        elif mode == "messages":
            chunk, metadata = data
            if (
                metadata.get("langgraph_node") == "generate"
                and hasattr(chunk, "content")
                and chunk.content
            ):
                yield ("token", chunk.content)

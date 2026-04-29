from typing import TypedDict

from langchain_core.documents import Document

# History is a list of (question, answer) pairs managed manually by the UI.
# TODO: replace with LangGraph MemorySaver or PostgresSaver for persistent
#       cross-session memory — swap is one line in build_graph().
History = list[tuple[str, str]]


class RAGState(TypedDict):
    question: str
    history: History
    standalone_question: str   # question rewritten to be self-contained given history
    hypothesis: str            # HyDE: hypothetical answer used as retrieval query
    documents: list[Document]
    answer: str

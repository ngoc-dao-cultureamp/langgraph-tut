from typing import TypedDict

from langchain_core.documents import Document

# History is managed by MemorySaver via thread_id — not stored in state manually.
# To persist across app restarts, swap MemorySaver for PostgresSaver in build_graph().


class RAGState(TypedDict):
    question: str
    history: str               # formatted prior turns, built from MemorySaver before invocation
    standalone_question: str   # question rewritten to be self-contained given history
    hypothesis: str            # HyDE: hypothetical answer used as retrieval query
    documents: list[Document]
    answer: str

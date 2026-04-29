from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    hypothesis: str        # HyDE: hypothetical answer used as retrieval query
    documents: list[Document]
    answer: str

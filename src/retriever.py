import logging

from langchain_core.documents import Document

from store import get_vector_store

logger = logging.getLogger(__name__)


def search(query: str, k: int = 10) -> list[Document]:
    try:
        return get_vector_store().similarity_search(query, k=k)
    except Exception as exc:
        logger.error("search() failed: %s", exc)
        raise RuntimeError("Could not search documents — database may be unavailable.") from exc


def list_all() -> list[Document]:
    store = get_vector_store()
    with store._make_sync_session() as session:
        collection = store.get_collection(session)
        if collection is None:
            return []
        records = (
            session.query(store.EmbeddingStore)
            .filter(store.EmbeddingStore.collection_id == collection.uuid)
            .order_by(store.EmbeddingStore.cmetadata["source"].astext)
            .all()
        )
    return [
        Document(page_content=r.document, metadata=r.cmetadata or {})
        for r in records
    ]

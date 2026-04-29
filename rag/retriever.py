import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DB_URL = os.environ["DB_URL"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "snowflake-arctic-embed2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = "docs"


def get_vector_store() -> PGVector:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def search(query: str, k: int = 10) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)


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

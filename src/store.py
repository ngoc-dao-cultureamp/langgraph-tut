import os

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

DB_URL = os.environ["DB_URL"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text-v1.5")
EMBED_HOST = os.environ.get("EMBED_HOST", "http://localhost:8081/v1")
COLLECTION_NAME = "docs"


def get_vector_store() -> PGVector:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, base_url=EMBED_HOST, api_key="sk-local", check_embedding_ctx_length=False)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )

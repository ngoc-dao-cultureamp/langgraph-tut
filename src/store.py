import logging
import os

import sqlalchemy
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

DB_URL = os.environ["DB_URL"]
EMBED_HOST = os.environ.get("EMBED_HOST")
EMBED_MODEL_ALIAS = os.environ.get("EMBED_MODEL_ALIAS")
COLLECTION_NAME = "docs"

logger = logging.getLogger(__name__)


def check_connection() -> None:
    """Ping postgres. Logs technical details and raises RuntimeError on failure."""
    try:
        engine = sqlalchemy.create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    except Exception as exc:
        logger.error("PostgreSQL unavailable at %s: %s", DB_URL, exc)
        raise RuntimeError(
            "Database is unavailable. Run `devbox services up` to start PostgreSQL."
        ) from exc


def get_vector_store() -> PGVector:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL_ALIAS, base_url=EMBED_HOST, api_key="sk-local", check_embedding_ctx_length=False)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )

import os

from dotenv import load_dotenv
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

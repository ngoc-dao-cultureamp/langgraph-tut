"""Ingest documents from DOCS_DIR into pgvector."""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DOCS_DIR = Path(os.environ["DOCS_DIR"])
DB_URL = os.environ["DB_URL"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "snowflake-arctic-embed2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = "docs"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def _parse_file(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if match:
        metadata = yaml.safe_load(match.group(1)) or {}
        content = match.group(2).strip()
    else:
        metadata = {}
        content = raw.strip()

    metadata["source"] = path.name
    # Normalise list fields to comma-separated strings for pgvector metadata
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ", ".join(str(v) for v in value)

    return Document(page_content=content, metadata=metadata)


def load_documents() -> list[Document]:
    docs = [_parse_file(p) for p in sorted(DOCS_DIR.glob("*.txt"))]
    return docs


def get_vector_store() -> PGVector:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def run(on_progress=None) -> int:
    """Ingest all docs. Calls on_progress(current, total) if provided. Returns doc count."""
    docs = load_documents()
    total = len(docs)

    store = get_vector_store()
    # Drop and recreate the collection so re-ingestion is always clean
    store.drop_tables()
    store.create_tables_if_not_exists()
    store.create_collection()

    batch_size = 32
    ingested = 0
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch)
        ingested += len(batch)
        if on_progress:
            on_progress(ingested, total)

    return ingested


if __name__ == "__main__":
    def _print_progress(current, total):
        print(f"  {current}/{total}", end="\r", flush=True)

    print(f"Loading documents from {DOCS_DIR} ...")
    count = run(on_progress=_print_progress)
    print(f"\nDone. Ingested {count} documents.")

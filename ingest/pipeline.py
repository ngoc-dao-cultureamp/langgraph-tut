"""Ingest documents from DOCS_DIR and books from BOOKS_DIR into pgvector."""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(os.environ["DOCS_DIR"])
BOOKS_DIR = Path(os.environ.get("BOOKS_DIR", ""))
DB_URL = os.environ["DB_URL"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "snowflake-arctic-embed2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = "docs"

# Gutenberg boilerplate markers
_GUTENBERG_START_RE = re.compile(r"\*{3} START OF THE PROJECT GUTENBERG EBOOK .+? \*{3}", re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r"\*{3} END OF THE PROJECT GUTENBERG EBOOK .+? \*{3}", re.IGNORECASE)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)

# 1000 chars ≈ 200-250 tokens. Smaller chunks give more precise retrieval;
# the 8192-token model limit is a ceiling, not a target.
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _parse_doc_file(path: Path) -> Document:
    """Load a pre-chunked doc with optional YAML frontmatter."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if match:
        metadata = yaml.safe_load(match.group(1)) or {}
        content = match.group(2).strip()
    else:
        metadata = {}
        content = raw.strip()

    metadata["source"] = path.name
    metadata.setdefault("topic", path.stem.replace("_", " ").title())
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ", ".join(str(v) for v in value)

    return Document(page_content=content, metadata=metadata)


def _parse_book_file(path: Path) -> list[Document]:
    """Load a Gutenberg plain-text book, strip boilerplate, and chunk it."""
    raw = path.read_text(encoding="utf-8")

    # Strip Gutenberg header
    start_match = _GUTENBERG_START_RE.search(raw)
    if start_match:
        raw = raw[start_match.end():]

    # Strip Gutenberg footer
    end_match = _GUTENBERG_END_RE.search(raw)
    if end_match:
        raw = raw[:end_match.start()]

    raw = raw.strip()

    title = path.stem.replace("_", " ").title()
    base_metadata = {
        "source": path.name,
        "title": title,
        "author": "Arthur Conan Doyle",
        "topic": title,
        "type": "book",
    }

    chunks = _SPLITTER.create_documents([raw], metadatas=[base_metadata])
    # Tag each chunk with its position so results are easier to orient
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def load_documents() -> list[Document]:
    docs = [_parse_doc_file(p) for p in sorted(DOCS_DIR.glob("*.txt"))]
    return docs


def load_books() -> list[Document]:
    if not BOOKS_DIR or not BOOKS_DIR.is_dir():
        return []
    chunks = []
    for path in sorted(BOOKS_DIR.glob("*.txt")):
        chunks.extend(_parse_book_file(path))
    return chunks


def get_vector_store() -> PGVector:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def run(on_progress=None) -> int:
    """Ingest all docs and books. Calls on_progress(current, total) if provided. Returns chunk count."""
    all_docs = load_documents() + load_books()
    total = len(all_docs)

    store = get_vector_store()
    # Drop and recreate the collection so re-ingestion is always clean
    store.drop_tables()
    store.create_tables_if_not_exists()
    store.create_collection()

    batch_size = 32
    ingested = 0
    for i in range(0, total, batch_size):
        batch = all_docs[i : i + batch_size]
        store.add_documents(batch)
        ingested += len(batch)
        if on_progress:
            on_progress(ingested, total)

    return ingested


if __name__ == "__main__":
    import time
    _start = time.time()

    def _print_progress(current, total):
        elapsed = time.time() - _start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        print(f"  {current}/{total}  {rate:.0f} chunks/s  ~{remaining:.0f}s remaining", end="\r", flush=True)

    print(f"Loading docs from {DOCS_DIR} ...")
    print(f"Loading books from {BOOKS_DIR} ...")
    count = run(on_progress=_print_progress)
    print(f"\nDone. Ingested {count} chunks in {time.time() - _start:.0f}s.")

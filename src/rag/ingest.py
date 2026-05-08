"""Ingest books from BOOKS_DIR into pgvector."""

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.store import get_vector_store

BOOKS_DIR = Path(os.environ["BOOKS_DIR"])

_GUTENBERG_START_RE = re.compile(r"\*{3} START OF THE PROJECT GUTENBERG EBOOK .+? \*{3}", re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r"\*{3} END OF THE PROJECT GUTENBERG EBOOK .+? \*{3}", re.IGNORECASE)

# 1000 chars ≈ 200-250 tokens. Smaller chunks give more precise retrieval;
# the 8192-token model limit is a ceiling, not a target.
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _parse_book_file(path: Path) -> list[Document]:
    """Load a Gutenberg plain-text book, strip boilerplate, and chunk it."""
    raw = path.read_text(encoding="utf-8")

    start_match = _GUTENBERG_START_RE.search(raw)
    if start_match:
        raw = raw[start_match.end():]

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
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def load_books() -> list[Document]:
    chunks = []
    for path in sorted(BOOKS_DIR.glob("*.txt")):
        chunks.extend(_parse_book_file(path))
    return chunks


def run(on_progress=None) -> int:
    """Ingest all books. Calls on_progress(current, total) if provided. Returns chunk count."""
    docs = load_books()
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
    import time
    _start = time.time()

    def _print_progress(current, total):
        elapsed = time.time() - _start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0
        print(f"  {current}/{total}  {rate:.0f} chunks/s  ~{remaining:.0f}s remaining", end="\r", flush=True)

    print(f"Loading books from {BOOKS_DIR} ...")
    count = run(on_progress=_print_progress)
    print(f"\nDone. Ingested {count} chunks in {time.time() - _start:.0f}s.")

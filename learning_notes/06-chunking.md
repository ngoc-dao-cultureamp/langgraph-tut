# Chunking

## Why chunking is needed

A vector embedding represents the *meaning* of a piece of text as a single float
vector. If you embed an entire book, you get one vector that averages the meaning of
everything — useless for retrieval. Chunking breaks the text into small, semantically
focused pieces so each vector captures a specific idea.

## Two kinds of input in this project

| Source | Format | Needs chunking? |
|---|---|---|
| Internal docs (`DOCS_DIR`) | Pre-chunked `.txt` with YAML frontmatter | No — one file = one chunk |
| Sherlock Holmes books (`BOOKS_DIR`) | Long-form Gutenberg plain text | Yes |

## RecursiveCharacterTextSplitter

LangChain's `RecursiveCharacterTextSplitter` splits on a hierarchy of separators:
`\n\n` (paragraphs) → `\n` (lines) → ` ` (words) → characters.
It tries the largest separator first and falls back only when a chunk would still be
too long. This keeps paragraphs intact wherever possible.

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
```

- **chunk_size=1000** — ~200–250 tokens per chunk. The embedding model's 8192-token
  limit is a ceiling, not a target. Smaller chunks have more focused meaning, so their
  vectors score higher on relevant queries. A chunk covering one scene or idea retrieves
  better than a chunk covering an entire chapter.
- **chunk_overlap=150** — the last 150 characters of each chunk are repeated at the
  start of the next. This prevents losing context when a sentence or idea spans a
  split boundary.

## Gutenberg boilerplate stripping

Project Gutenberg files include a long legal header and footer. These are stripped
before chunking using the standard markers:

```
*** START OF THE PROJECT GUTENBERG EBOOK <title> ***
*** END OF THE PROJECT GUTENBERG EBOOK <title> ***
```

Without stripping, the boilerplate would end up in the vector store and pollute
retrieval results with licensing text.

## Chunk metadata

Each book chunk stores:
- `source` — filename
- `title` — book title
- `author` — Arthur Conan Doyle
- `topic` — same as title (used by the UI for display)
- `type` — "book" (distinguishes from internal docs)
- `chunk_index` / `chunk_total` — position in the book, useful for debugging

## Chunk count

A typical Sherlock Holmes book (~300–600 KB) produces ~400–800 chunks at chunk_size=1000.
The full 9-book corpus produces several thousand chunks on top of the 316 internal docs.

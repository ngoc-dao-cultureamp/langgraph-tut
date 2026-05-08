# Chunking

## Why chunking is needed

A vector embedding represents the *meaning* of a piece of text as a single float
vector. If you embed an entire book, you get one vector that averages the meaning of
everything — useless for retrieval. Chunking breaks the text into small, semantically
focused pieces so each vector captures a specific idea.

## Two kinds of input

RAG pipelines commonly handle two types of source material:

| Source | Format | Chunking needed? |
|---|---|---|
| Pre-structured docs | Short files, one topic per file | No — one file = one chunk |
| Long-form text (books, reports, transcripts) | Continuous prose | Yes |

For pre-structured docs, the file boundary is already a meaningful semantic unit. For long-form text, you must split explicitly.

## RecursiveCharacterTextSplitter

LangChain's `RecursiveCharacterTextSplitter` splits on a hierarchy of separators:
`\n\n` (paragraphs) → `\n` (lines) → ` ` (words) → characters.
It tries the largest separator first and falls back only when a chunk would still be
too long. This keeps paragraphs intact wherever possible.

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
```

- **chunk_size=1000** — ~200–250 tokens per chunk (assuming ~4 chars/token). The embedding model's context limit is a ceiling, not a target. Smaller chunks have more focused meaning, so their vectors score higher on relevant queries. A chunk covering one scene or idea retrieves better than a chunk covering an entire chapter.
- **chunk_overlap=150** — the last 150 characters of each chunk are repeated at the start of the next. This prevents losing context when a sentence or idea spans a split boundary.

## Boilerplate stripping

Many long-form sources include headers, footers, or legal text that is irrelevant to the content (e.g. Project Gutenberg licensing blocks, document cover pages). Strip these before chunking — otherwise they end up in the vector store and pollute retrieval results.

Use reliable start/end markers when available:

```
*** START OF THE PROJECT GUTENBERG EBOOK <title> ***
*** END OF THE PROJECT GUTENBERG EBOOK <title> ***
```

For other sources, strip by regex, line count, or structural cues (e.g. first `<body>` tag in HTML).

## Chunk metadata

Store metadata alongside each chunk to support filtering and debugging:
- **Source** — filename or URL
- **Title / author** — document identity
- **Topic / type** — used for UI display and metadata filtering
- **Chunk index / total** — position in the source document, useful for debugging retrieval

Metadata is stored in pgvector's `cmetadata` JSONB column and can be filtered in queries.

## Expected chunk counts

A rough guide for prose text at chunk_size=1000:
- ~300–600 KB source file → ~400–800 chunks
- Plan for several thousand chunks across a full corpus

Actual counts vary with paragraph density and overlap settings.

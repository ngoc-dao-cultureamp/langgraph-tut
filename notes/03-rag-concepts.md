# RAG Concepts

## What is RAG?

Retrieval-Augmented Generation (RAG) grounds an LLM's answers in a specific document
corpus. Instead of relying on the model's training data, you retrieve relevant passages
at query time and include them in the prompt.

```
question
   ↓
embed question → similarity search → top-k chunks
                                          ↓
                               stuff into prompt → LLM → answer
```

## Why RAG over fine-tuning?

| | RAG | Fine-tuning |
|---|---|---|
| Update knowledge | Re-ingest docs | Retrain model |
| Cost | Low | High |
| Hallucination risk | Lower (grounded) | Higher |
| Best for | Specific doc corpora | Changing model behaviour/style |

## The two pipelines

### Ingestion (offline, run once or on demand)

```
raw docs → load → chunk → embed → upsert into pgvector
```

- **Load**: read files (PDF, Markdown, plain text, HTML)
- **Chunk**: split into overlapping pieces so no context is lost at boundaries
- **Embed**: convert each chunk to a float vector using an embedding model
- **Upsert**: store vector + original text + metadata in pgvector

The LLM is **not** involved here — only the embedding model.

### Query (online, per user question)

```
question → embed → similarity search → retrieve top-k → LLM → answer
```

- **Embed**: convert the question to a vector using the same embedding model
- **Similarity search**: find the closest vectors in pgvector (`<=>` cosine distance)
- **Retrieve**: return the original text of the top-k chunks
- **Generate**: LLM reads the chunks + question and writes an answer

## Chunking strategy

Chunks should be:
- Small enough to be specific (avoid stuffing unrelated content)
- Large enough to be self-contained (avoid cutting mid-thought)
- Overlapping slightly so context isn't lost at split boundaries

`snowflake-arctic-embed2` supports 8192-token context, so chunks can be generous.
A good starting point: 1000 tokens, 200 token overlap.

In this project the docs are already pre-chunked (each `.txt` file is one chunk),
so no further splitting is needed during ingestion.

## Agentic RAG with LangGraph

Plain RAG is a single-pass chain. LangGraph adds control flow:

```
question → retrieve → grade docs → good? → generate → answer
                           ↓ bad?
                      rewrite query → retrieve again
```

Nodes you can add:
- **Grade documents** — check if retrieved chunks are actually relevant
- **Query rewriter** — rephrase the question if retrieval quality is poor
- **Web search fallback** — route to a search tool if the corpus has no answer
- **Hallucination checker** — verify the answer is grounded in the retrieved docs

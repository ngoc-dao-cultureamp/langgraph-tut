# HyDE — Hypothetical Document Embeddings

## The problem it solves

Standard RAG embeds the *question* and searches for similar vectors. But questions and
answers live in different semantic spaces:

- Question: "Where does Sherlock Holmes live?"
- Answer in corpus: *"You had better come round and see my rooms in Baker Street..."*

These don't look alike to an embedding model. HyDE bridges that gap.

## How it works

Instead of embedding the question directly, ask the LLM to generate a *hypothetical
answer*, then embed and search that:

```
question
   ↓ LLM
"Sherlock Holmes lives at 221B Baker Street, London, where he shares rooms with Dr. Watson."
   ↓ embed
pgvector similarity search
   ↓
real chunks about Baker Street
```

The hypothetical answer looks like corpus text — prose matching prose — so it retrieves
better.

## Does the LLM need to already know the answer?

**Partly.** Two cases:

**Well-known domain (e.g. Sherlock Holmes):** The LLM was trained on this text and
knows the answer. The hypothetical is accurate and retrieval is precise.

**Private/unknown domain (e.g. internal docs):** The LLM guesses. But a plausible-
sounding guess in the right *style and vocabulary* still retrieves better than the raw
question. The style match matters more than factual accuracy for retrieval purposes.

## The tradeoff

| | Standard retrieval | HyDE |
|---|---|---|
| Failure mode | Semantic mismatch (question ≠ answer style) | Hallucinated hypothetical misdirects search |
| Best for | Short factual queries | Questions where LLM has domain knowledge |
| Worst for | Long narrative questions | Fully private corpora |

If the LLM confidently invents a wrong detail — a quote, a name, a location — the
hypothetical vector steers retrieval toward chunks that match the invention, not the
truth.

## In LangGraph

HyDE adds one node before retrieval:

```
START → hypothesize → retrieve → generate → END
```

The `hypothesize` node calls the LLM with a prompt like:

```
Write a short passage that would answer this question if it appeared in the source text.
Question: {question}
```

The output replaces the original question as the retrieval query.

## Performance: disable thinking for intermediate steps

With a thinking model like Qwen3, the `hypothesize` node (and `filter`, `rewrite`) are
slow because the model silently generates a `<think>` block before answering —
even though that reasoning is never surfaced or used.

Fix: pass `enable_thinking=False` via `extra_body` on the `ChatOpenAI` call:

```python
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
llm = ChatOpenAI(..., extra_body=_NO_THINK)
```

The model skips the think block entirely and responds immediately. Only the final
`generate` step (which uses the raw openai client and surfaces thinking to the user)
should keep thinking enabled.

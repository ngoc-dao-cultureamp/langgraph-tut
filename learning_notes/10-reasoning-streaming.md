# Reasoning streaming (thinking tokens)

## What are thinking tokens?

Qwen3 is a *thinking* model. Before writing its final answer it reasons through
the problem internally, producing a separate stream of tokens called
`reasoning_content`. These are stripped from the final answer but are visible
while streaming — they let you see the model's chain-of-thought as it happens.

## How llama.cpp exposes them

llama.cpp surfaces thinking tokens as an extra field on each streaming delta:

```json
{"choices": [{"delta": {"role": "assistant", "content": "", "reasoning_content": "Let me think..."}}]}
{"choices": [{"delta": {"content": "The answer is 42."}}]}
```

No special flags are needed — llama.cpp emits `reasoning_content` automatically
for models that support thinking (like Qwen3).

## Why LangChain can't be used for this

`ChatOpenAI` from `langchain-openai` reassembles streaming deltas using only the
`content` field. `reasoning_content` is silently discarded before it ever reaches
your code.

To verify: add a `print(chunk)` inside a LangChain streaming loop — you'll see
only the answer tokens, never the thinking tokens.

## The fix: raw openai client

Use the `openai` Python client directly. It preserves all fields on the delta:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

for chunk in client.chat.completions.create(model="qwen3", messages=messages, stream=True):
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print("[thinking]", delta.reasoning_content, end="")   # thinking token
    if delta.content:
        print(delta.content, end="")                           # answer token
```

`getattr(..., None)` guards against the field being absent on non-thinking models
(or on chunks that only carry `content`).

## Event protocol in this project

`llm.py` wraps the raw client and yields typed 2-tuples so callers don't need to
inspect delta fields directly:

```python
type StreamEvent = tuple[str, str]

("thinking", "Let me think...")   # one reasoning token chunk
("token",    "The answer is 42.") # one answer token chunk
```

`graph.py::stream_answer` extends this with pipeline events:

```python
("standalone", "Who is Sherlock Holmes?")  # rewritten question
("hypothesis", "Holmes lived at 221B...")  # HyDE passage
("docs",       [Document, ...])            # retrieved chunks
("thinking",   "...")                      # reasoning token
("token",      "...")                      # answer token
```

Consumers (CLI, Streamlit UI) switch on the event type and render accordingly.

## Architecture decision

LangGraph is used for the retrieval pipeline (filter → rewrite → hypothesize →
retrieve). But `_generate` is **not** a graph node — it runs manually after the
graph completes, using the raw openai client.

This keeps the reasoning stream intact while still benefiting from LangGraph's
stateful checkpointing for multi-turn context.

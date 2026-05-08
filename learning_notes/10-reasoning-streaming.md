# Reasoning streaming (thinking tokens)

## What are thinking tokens?

Some LLMs have a *thinking* or *reasoning* mode where the model reasons through
the problem before producing a final answer. This internal reasoning is emitted
as a separate token stream — often called `reasoning_content` or `thinking` —
that is distinct from the visible answer.

This is useful for:
- Seeing the model's chain-of-thought as it happens
- Debugging why the model reached a particular answer
- Gauging confidence (shallow reasoning → less reliable answer)

Models that support this include Qwen3, DeepSeek-R1, and o1/o3-family models.
The exact field name and API shape varies by provider.

## How llama.cpp exposes them

llama.cpp (and compatible servers like vLLM) surface thinking tokens as an extra
field on each streaming delta:

```json
{"choices": [{"delta": {"role": "assistant", "content": "", "reasoning_content": "Let me think..."}}]}
{"choices": [{"delta": {"content": "The answer is 42."}}]}
```

The field name is `reasoning_content`. For OpenAI o1/o3, the equivalent appears
in `message.reasoning` (non-streaming) or may be embedded in the content with
special tags depending on the API version.

## Disabling reasoning per-call

For models served via llama.cpp, reasoning can be disabled per-request:

```python
extra = {"chat_template_kwargs": {"enable_thinking": False}}
client.chat.completions.create(..., extra_body=extra)
```

Other providers/servers use different knobs:
- llama.cpp: `chat_template_kwargs.enable_thinking`
- Some models: a system prompt directive like `/no_think`
- OpenAI o-series: reasoning cannot be disabled but `reasoning_effort` controls depth

**When to disable:** any intermediate pipeline step where the output is short and
deterministic (classification, rewriting, passage generation). These don't benefit
from deep reasoning and the hidden think block wastes time and tokens.

## Why LangChain can't stream thinking tokens

`ChatOpenAI` from `langchain-openai` reassembles streaming deltas using only the
`content` field. Extra fields like `reasoning_content` are silently discarded
before they reach your code. This is a general LangChain limitation — it only
forwards fields it knows about.

The same issue would apply to any non-standard field a provider adds to the delta.

## The fix: raw openai client

Use the `openai` Python client directly. It returns the raw delta object and
preserves all fields:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")

for chunk in client.chat.completions.create(model="...", messages=messages, stream=True):
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):  # field absent on non-thinking models
        print("[thinking]", delta.reasoning_content, end="")
    if delta.content:
        print(delta.content, end="")
```

`getattr(..., None)` makes the code safe for models that don't emit thinking tokens.

## Event protocol in this project

`graph/llm.py` wraps the raw client and yields typed 2-tuples:

```python
type StreamEvent = tuple[str, str]

("thinking", "Let me think...")   # one reasoning token chunk
("token",    "The answer is 42.") # one answer token chunk
```

`graph/graph.py::stream_answer` extends this with RAG pipeline events:

```python
("standalone", "Who is Sherlock Holmes?")  # rewritten question
("hypothesis", "Holmes lived at 221B...")  # HyDE passage
("docs",       [Document, ...])            # retrieved chunks
("thinking",   "...")                      # reasoning token
("token",      "...")                      # answer token
```

## Architecture decision

LangGraph is used for the retrieval pipeline (filter → rewrite → hypothesize →
retrieve). The generate step runs **outside** the graph, directly against the LLM
client, so the reasoning stream is preserved end-to-end.

This pattern — LangGraph for structured retrieval, raw client for generation —
applies any time you need access to non-standard streaming fields that LangChain
would otherwise drop.

# 08 — Guardrails: Topic Filter with Conditional Edge

A guardrail is a node that can short-circuit the graph before expensive work happens.

## The pattern

```
START → filter ──yes──→ rewrite → hypothesize → retrieve → generate → END
                └──no──→ END
```

The `filter` node asks a cheap LLM call: "is this question on-topic, yes or no?"
If off-topic, it sets `answer` directly and returns to `END`, skipping retrieval and generation entirely.

## Why a conditional edge, not an if/else inside a node

A node can only produce state — it cannot decide where the graph goes next. Routing is always done by a **conditional edge**: a function that reads state and returns a string key that maps to the next node.

```python
def _is_off_topic(state: RAGState) -> str:
    return "off_topic" if state.get("answer") else "on_topic"

g.add_conditional_edges("filter", _is_off_topic, {"off_topic": END, "on_topic": "rewrite"})
```

This separation (node = data transform, edge = routing decision) is a core LangGraph design principle. It keeps nodes pure and makes the graph topology inspectable.

## Surfacing the off-topic answer to the UI

The off-topic answer is set in `state["answer"]` by the `filter` node, not streamed token-by-token from `generate`. So `stream_answer()` must check the `updates` stream for it:

```python
if "filter" in data and data["filter"].get("answer"):
    yield ("token", data["filter"]["answer"])
```

This means the UI gets a single `("token", "I can only answer...")` event instead of incremental chunks — fine for a short canned response.

## Tradeoffs

- **Latency**: adds one LLM call per turn, even for on-topic questions
- **False positives**: "What year was Watson born?" might get a "no" from a weak model
- **Easy to bypass**: not a security boundary, just a UX guardrail
- **Worth it when**: you expose the app to untrusted users or want clean UX for a focused app

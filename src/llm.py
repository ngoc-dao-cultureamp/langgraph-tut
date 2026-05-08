"""Low-level LLM streaming helpers using the raw openai client.

These bypass LangChain's ChatOpenAI so that reasoning_content (Qwen3 thinking
tokens) is preserved in the stream — LangChain silently drops that field.
"""

import os
from collections.abc import Generator

from openai import OpenAI

LLM_HOST = os.environ.get("LLM_HOST", "http://localhost:8080/v1")
LLM_MODEL_ALIAS = os.environ.get("LLM_MODEL_ALIAS", "")

type StreamEvent = tuple[str, str]


def stream_free(
    messages: list[dict],
    enable_reasoning: bool = True,
) -> Generator[StreamEvent, None, None]:
    """Stream a free (non-RAG) chat turn.

    messages: full conversation as [{"role": ..., "content": ...}, ...]
              with the new user message already appended.
    enable_reasoning: set False to skip the <think> block (faster, no reasoning tokens).

    Yields:
      ("thinking", str) — reasoning token (only when enable_reasoning=True)
      ("token",   str)  — answer token
    """
    extra = {"chat_template_kwargs": {"enable_thinking": enable_reasoning}}
    client = OpenAI(base_url=LLM_HOST, api_key="sk-local")
    for chunk in client.chat.completions.create(
        model=LLM_MODEL_ALIAS,
        messages=messages,
        stream=True,
        extra_body=extra,
    ):
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning_content", None):
            yield ("thinking", delta.reasoning_content)
        if delta.content:
            yield ("token", delta.content)

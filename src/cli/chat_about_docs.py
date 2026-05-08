"""CLI for RAG chat — mirrors the "Chat about Docs" tab in the web UI.

Useful for verifying the full LangGraph pipeline end-to-end:
  filter → rewrite → hypothesize → retrieve → generate (with reasoning).

Requires PostgreSQL and llama-server to be running:
    devbox services up

Usage:
    uv run python src/cli/chat_about_docs.py              # reasoning on (default)
    uv run python src/cli/chat_about_docs.py --no-reasoning  # reasoning off

Intermediate graph steps (standalone question, hypothesis, doc count) are
printed in dim grey. Reasoning tokens are printed in dim grey. Answer tokens
are printed normally. Press Ctrl-C or Ctrl-D to exit.
"""

import argparse
import sys
import uuid

from graph.graph import build_graph, stream_answer

RESET  = "\033[0m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"

compiled_graph = build_graph()
thread_id = str(uuid.uuid4())
enable_reasoning: bool = True


def chat(question: str) -> None:
    in_thinking = False
    in_answer = False
    print()
    try:
        for event, payload in stream_answer(question, compiled_graph, thread_id, enable_reasoning=enable_reasoning):
            if event == "standalone":
                print(f"{DIM}[standalone] {payload}{RESET}", flush=True)
            elif event == "hypothesis":
                print(f"{DIM}[hypothesis] {payload}{RESET}", flush=True)
            elif event == "docs":
                print(f"{DIM}[retrieved {len(payload)} chunks]{RESET}", flush=True)
            elif event == "thinking":
                if not in_thinking:
                    print(f"{DIM}[thinking]{RESET}", flush=True)
                    in_thinking = True
                print(f"{DIM}{payload}{RESET}", end="", flush=True)
            elif event == "token":
                if in_thinking:
                    print(f"\n{RESET}", flush=True)
                    in_thinking = False
                if not in_answer:
                    print()
                    in_answer = True
                print(payload, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


def main() -> None:
    global enable_reasoning
    parser = argparse.ArgumentParser(description="Docs chat CLI")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning tokens")
    args = parser.parse_args()
    enable_reasoning = not args.no_reasoning
    mode = "reasoning OFF" if args.no_reasoning else "reasoning ON"
    print(f"{BOLD}Docs chat CLI{RESET} (thread {thread_id[:8]}…, {mode}) — Ctrl-C or Ctrl-D to exit\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not question:
            continue
        chat(question)


if __name__ == "__main__":
    main()

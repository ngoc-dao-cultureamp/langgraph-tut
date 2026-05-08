"""CLI for free chat — mirrors the "Chat about anything" tab in the web UI.

Useful for verifying that reasoning/thinking streaming works end-to-end.

Usage:
    uv run python src/cli_chat_about_anything.py           # reasoning on (default)
    uv run python src/cli_chat_about_anything.py --no-think  # reasoning off

Reasoning tokens are printed dimmed in grey; answer tokens are printed normally.
Press Ctrl-C or Ctrl-D to exit.
"""

import argparse
import sys

from llm import stream_free

RESET = "\033[0m"
DIM   = "\033[2m"
BOLD  = "\033[1m"

messages: list[dict] = []
enable_thinking: bool = True


def chat(question: str) -> None:
    messages.append({"role": "user", "content": question})
    answer = ""
    in_thinking = False
    print()
    try:
        for event, payload in stream_free(messages, enable_thinking=enable_thinking):
            if event == "thinking":
                if not in_thinking:
                    print(f"{DIM}[thinking]{RESET}", flush=True)
                    in_thinking = True
                print(f"{DIM}{payload}{RESET}", end="", flush=True)
            elif event == "token":
                if in_thinking:
                    print(f"\n{RESET}", flush=True)
                    in_thinking = False
                print(payload, end="", flush=True)
                answer += payload
    except KeyboardInterrupt:
        pass
    print()
    messages.append({"role": "assistant", "content": answer})


def main() -> None:
    global enable_thinking
    parser = argparse.ArgumentParser(description="Free chat CLI")
    parser.add_argument("--no-think", action="store_true", help="Disable reasoning tokens")
    args = parser.parse_args()
    enable_thinking = not args.no_think
    mode = "reasoning OFF" if args.no_think else "reasoning ON"
    print(f"{BOLD}Free chat CLI{RESET} ({mode}) — Ctrl-C or Ctrl-D to exit\n")
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

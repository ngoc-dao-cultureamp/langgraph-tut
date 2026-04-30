"""CLI: ask a question and print JSON output.

Usage:
    uv run python src/ask.py "Where does Sherlock Holmes live?"
"""

import json
import sys

from graph import build_graph, stream_answer


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python src/ask.py <question>", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    graph = build_graph()
    thread_id = "cli"

    result = {
        "question": question,
        "standalone": "",
        "hypothesis": "",
        "documents": [],
        "answer": "",
    }

    for event, payload in stream_answer(question, graph, thread_id):
        if event == "standalone":
            result["standalone"] = payload
        elif event == "hypothesis":
            result["hypothesis"] = payload
        elif event == "docs":
            result["documents"] = [
                {
                    "title": doc.metadata.get("title", ""),
                    "chunk_index": doc.metadata.get("chunk_index", ""),
                    "chunk_total": doc.metadata.get("chunk_total", ""),
                    "content": doc.page_content,
                }
                for doc in payload
            ]
        elif event == "token":
            result["answer"] += payload

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

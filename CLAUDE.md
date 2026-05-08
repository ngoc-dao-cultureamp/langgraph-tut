# Claude Instructions

## Learning notes folder

When explaining concepts, architecture decisions, debugging discoveries, or gotchas that would help an AI/RAG learner understand the "why", add or update a Markdown file in `learning_notes/`.

- Use number prefixes to maintain reading order (`01-`, `02-`, etc.)
- Keep notes concise and example-driven
- Add a new file when the topic is distinct; update an existing one when it's an extension

## Project context

- User is learning RAG and LangGraph — explain the "why" behind decisions, not just the "what"
- Devbox is the only package manager on this Mac — do not suggest Homebrew or Docker
- `uv` manages Python dependencies — do not use `pip` directly
- Services are started with `devbox services up` (process-compose) — do not suggest running postgres or llama-server manually

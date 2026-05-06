# Stack

## Local development

| Concern | Tool | Why |
|---|---|---|
| Package management | Devbox (Nix) | Reproducible environment across machines |
| Python deps | uv | Fast, standard `pyproject.toml`, generates lockfile |
| PostgreSQL + pgvector | Devbox Nix flake | pgvector must be compiled into PostgreSQL; Nix flake handles this |
| LLM inference | llama.cpp | Direct GGUF runtime; explicit control over context size, pooling, quantization |
| Vector store | pgvector | PostgreSQL extension; no separate vector DB needed |
| Graph / RAG logic | LangGraph | Stateful agent graphs with explicit control flow |
| UI | Streamlit | Richer than Chainlit once you need buttons alongside chat |

## Why llama.cpp over Ollama

Ollama wraps llama.cpp but adds a daemon layer that manages model loading behind the scenes — useful for a general-purpose desktop tool, but a source of surprises in a dev environment (unexpected model swaps, hidden context limits). Running llama-server directly gives full control and one fewer moving part.

Both expose an OpenAI-compatible API, so the Python code (`langchain-openai`) is the same either way.

## AWS deployment (future)

| Concern | Tool | Change from local |
|---|---|---|
| LLM | AWS Bedrock | Swap `ChatOpenAI` → `ChatBedrock` via `LLM_PROVIDER` env var |
| PostgreSQL | RDS PostgreSQL | Same pgvector extension, managed by AWS |
| App hosting | App Runner | Containerised Streamlit, auto-scales |

## Services

Started locally with `devbox services up` via process-compose:

- **postgres** — PostgreSQL 17 with pgvector, data in `.pgdata/`
- **llama-chat** — LLM inference on port 8080, model file in `models/`
- **llama-embed** — Embedding inference on port 8081, model file in `models/`
- **open-webui** — Chat UI on port 3000, backed by llama-chat

# Stack

## Local development

| Concern | Tool | Why |
|---|---|---|
| Package management | Devbox (Nix) | Reproducible environment across machines |
| Python deps | uv | Fast, standard `pyproject.toml`, generates lockfile |
| PostgreSQL + pgvector | Devbox Nix flake | pgvector must be compiled into PostgreSQL; Nix flake handles this |
| LLM inference | Ollama | Single tool for both Mac (Metal) and Linux/CUDA; same API everywhere |
| Vector store | pgvector | PostgreSQL extension; no separate vector DB needed |
| Graph / RAG logic | LangGraph | Stateful agent graphs with explicit control flow |
| UI | Streamlit | Richer than Chainlit once you need buttons alongside chat |

## AWS deployment (future)

| Concern | Tool | Change from local |
|---|---|---|
| LLM | AWS Bedrock | Swap `ChatOllama` → `ChatBedrock` via `LLM_PROVIDER` env var |
| PostgreSQL | RDS PostgreSQL | Same pgvector extension, managed by AWS |
| App hosting | App Runner | Containerised Streamlit, auto-scales |

## Services

Started locally with `devbox services up` via process-compose:

- **postgres** — PostgreSQL 17 with pgvector, data in `.pgdata/`
- **ollama** — LLM inference server, models stored in `~/.ollama/`

# Stack

## Local development

| Concern | Tool | Why |
|---|---|---|
| Package management | Devbox (Nix) | Reproducible environment across machines |
| Python deps | uv | Fast, standard `pyproject.toml`, generates lockfile |
| PostgreSQL + pgvector | Devbox Nix flake | pgvector must be compiled into PostgreSQL; Nix flake handles this |
| LLM inference | llama-server (self-compiled llama.cpp) | Full control over build flags; CUDA support compiled in at build time |
| Vector store | pgvector | PostgreSQL extension; no separate vector DB needed |
| Graph / RAG logic | LangGraph | Stateful agent graphs with explicit control flow |
| UI | Streamlit | Richer than Chainlit once you need buttons alongside chat |

## Why llama.cpp over Ollama

Ollama wraps llama.cpp but adds a daemon layer that manages model loading behind the scenes — useful for a general-purpose desktop tool, but a source of surprises in a dev environment (unexpected model swaps, hidden context limits). Running llama-server directly gives full control and one fewer moving part.

Both expose an OpenAI-compatible API, so the Python code (`langchain-openai`) is the same either way.

## Why self-compiled llama.cpp instead of llama-cpp-python

We use the `llama-server` binary from a self-compiled llama.cpp build rather than the `llama-cpp-python` Python wheel. The main reason is **CUDA support**:

- The `llama-cpp-python` wheels on the custom index (`abetlen.github.io/llama-cpp-python/whl/`) lag behind upstream llama.cpp and are not consistently published for every CUDA version.
- The standalone `llama-server` binaries in the llama.cpp GitHub releases are also inconsistently built for CUDA.
- Building from source is the only reliable way to get a CUDA-enabled binary that matches your driver and toolkit.

The `llama-server` binary exposes an OpenAI-compatible HTTP API (`/v1`) — identical to what `llama-cpp-python` served. No Python code changes are needed; the LangChain `ChatOpenAI` client talks to the same endpoints.

The path to the binary is configured in `local.env` (gitignored) via `LLAMA_SERVER_BIN`. `devbox shell` aborts with a clear error if this variable is not set.

## AWS deployment (future)

| Concern | Tool | Change from local |
|---|---|---|
| LLM | AWS Bedrock | Swap `ChatOpenAI` → `ChatBedrock` via `LLM_PROVIDER` env var |
| PostgreSQL | RDS PostgreSQL | Same pgvector extension, managed by AWS |
| App hosting | App Runner | Containerised Streamlit, auto-scales |

## Services

Started locally with `devbox services up` via process-compose:

- **postgres** — PostgreSQL 17 with pgvector, data in `.pgdata/`
- **llama-chat** — LLM inference on port 8080, model file in `models/`; built-in chat UI at http://localhost:8080
- **llama-embed** — Embedding inference on port 8081, model file in `models/`

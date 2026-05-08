# LangGraph RAG Tutorial

A minimal RAG app using LangGraph, PostgreSQL + pgvector, llama.cpp, and Streamlit.

## Prerequisites

- [Devbox](https://www.jetify.com/devbox)

## Setup

### 1. Enter the devbox shell

```bash
devbox shell
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Download models (first time only)

```bash
devbox run model-pull
```

Downloads `$LLM_FILE` (`$LLM_REPO`) and `$EMBED_FILE` (`$EMBED_REPO`) into `models/`.

`$LLM_MODEL_ALIAS` runs well on both M3 Max (36GB) and RTX 3090 (24GB VRAM).

### 4. Start services

```bash
devbox services up
```

Starts PostgreSQL, two llama-server instances, and Open WebUI. On first run, PostgreSQL is automatically initialised and the `vector` extension enabled.

### 5. Ingest documents

```bash
uv run python src/ingest.py
```

## Services

| Command | Description |
|---|---|
| `devbox services up` | Start PostgreSQL, llama-server (chat + embed), and Open WebUI |
| `devbox services down` | Stop all services |
| `devbox run db-reset` | Wipe and reinitialise the database |
| `devbox run model-pull` | Download embedding model + LLM (first time only) |

### llama-server ports

| Service | Port | Purpose |
|---|---|---|
| `llama-chat` | `8080` | LLM (`$LLM_MODEL_ALIAS`) — chat completions |
| `llama-embed` | `8081` | Embeddings (`$EMBED_MODEL_ALIAS`) |

Both expose an OpenAI-compatible API (`/v1`). `llama-chat` also serves a built-in chat UI at **http://localhost:8080** — open it in a browser for quick ad-hoc prompts without the RAG pipeline.

## Connection details

| Setting | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| User | `$PGUSER` |
| Database | `$PGDATABASE` |
| URL | `$DB_URL` |

## Stack

| Concern | Tool |
|---|---|
| Python + app tooling | Devbox |
| PostgreSQL + pgvector | Devbox (Nix flake) |
| LLM inference (local) | llama.cpp |
| LLM inference (AWS) | AWS Bedrock |
| Vector store | pgvector |
| Graph / RAG logic | LangGraph |
| UI | Streamlit |
| Reference chat UI | Open WebUI |

# LangGraph RAG Tutorial

A minimal RAG app using LangGraph, PostgreSQL + pgvector, Ollama, and Streamlit.

## Prerequisites

- [Devbox](https://www.jetify.com/devbox)

## Setup

### 1. Enter the devbox shell

```bash
devbox shell
```

### 2. Install Python dependencies

```bash
pip install -e .
```

### 3. Copy and configure environment variables

```bash
cp .env.example .env
# Edit .env if your paths or ports differ
```

### 4. Pull Ollama models (first time only)

```bash
devbox run ollama-pull
```

This pulls `snowflake-arctic-embed2` (embedding model) and `qwen2.5:32b` (LLM).

`qwen2.5:32b` runs well on both M3 Max (36GB) and RTX 3090 (24GB VRAM).

### 5. Start services

```bash
devbox services up
```

Starts PostgreSQL (with pgvector) and Ollama together. On first run, PostgreSQL is automatically initialised and the `vector` extension enabled.

### 6. Ingest documents

```bash
python -m ingest.pipeline
```

## Services

| Command | Description |
|---|---|
| `devbox services up` | Start PostgreSQL + Ollama |
| `devbox services down` | Stop all services |
| `devbox run db-reset` | Wipe and reinitialise the database |
| `devbox run ollama-pull` | Pull embedding model + LLM (first time only) |

## Connection details

| Setting | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| User | `postgres` |
| Database | `ragdb` |
| URL | `postgresql://postgres@localhost:5432/ragdb` |

## Stack

| Concern | Tool |
|---|---|
| Python + app tooling | Devbox |
| PostgreSQL + pgvector | Devbox (Nix flake) |
| LLM inference (local) | Ollama |
| LLM inference (AWS) | AWS Bedrock |
| Vector store | pgvector |
| Graph / RAG logic | LangGraph |
| UI | Streamlit |

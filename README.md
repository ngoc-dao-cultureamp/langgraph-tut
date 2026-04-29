# LangGraph RAG Tutorial

A minimal RAG app using LangGraph, PostgreSQL + pgvector, Ollama, and Streamlit.

## Prerequisites

- [Devbox](https://www.jetify.com/devbox)

## Setup

### 1. Enter the devbox shell

```bash
devbox shell
```

### 2. Initialize the database (first time only)

```bash
devbox run db-init
```

This creates a local PostgreSQL 17 instance in `.pgdata/`, creates the `ragdb` database, and enables the `pgvector` extension.

### 3. Pull Ollama models (first time only)

```bash
devbox run ollama-pull
```

This pulls `snowflake-arctic-embed2` (embedding model) and `qwen2.5:32b` (LLM).

`qwen2.5:32b` runs well on both M3 Max (36GB) and RTX 3090 (24GB VRAM).

### 4. Start the database and Ollama

```bash
devbox run db-start
devbox run ollama-start
```

## Database commands

| Command | Description |
|---|---|
| `devbox run db-init` | Initialize and create DB (first time only) |
| `devbox run db-start` | Start PostgreSQL |
| `devbox run db-stop` | Stop PostgreSQL |
| `devbox run db-status` | Check if PostgreSQL is running |

## Connection details

| Setting | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| User | `postgres` |
| Database | `ragdb` |
| URL | `postgresql://postgres@localhost:5432/ragdb` |

## Ollama commands

| Command | Description |
|---|---|
| `devbox run ollama-start` | Start Ollama server in background |
| `devbox run ollama-pull` | Pull embedding model + LLM (first time only) |

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

# Devbox, PostgreSQL, and pgvector

## Why a Nix flake is needed for pgvector

`devbox add postgresql` works fine, but pgvector must be compiled *into* a specific
PostgreSQL build. Devbox's simple `devbox add` syntax doesn't expose Nix's
`withPackages` composition. A local `flake.nix` is needed to express:

```nix
pkgs.postgresql_17.withPackages (p: [ p.pgvector ])
```

The flake in this repo (`flake.nix`) supports all four architectures:
`aarch64-darwin`, `x86_64-darwin`, `x86_64-linux`, `aarch64-linux`.
Devbox references it in `devbox.json` as `"path:.#default"`.

## Database layout

All data lives in `.pgdata/` inside the project directory (gitignored).
PostgreSQL is started as a local process — no Docker, no system daemon.

On first `devbox services up`, the `postgres` process-compose service:
1. Runs `initdb` to create the cluster
2. Starts the server
3. Creates the `ragdb` database
4. Enables the `vector` extension

Subsequent starts detect the existing cluster and skip init.

## Useful psql commands

```bash
# Connect to the database
psql -U postgres -d ragdb

# Inside psql:
\dt                    -- list tables
\d langchain_pg_embedding  -- describe embedding table schema
SELECT count(*) FROM langchain_pg_embedding;
```

## Resetting the database

```bash
devbox run db-reset
```

This stops PostgreSQL, wipes `.pgdata/`, and reinitialises from scratch.
Run `uv run python -m ingest.pipeline` afterwards to re-ingest documents.

# pgvector Schema

`langchain_postgres` creates two tables automatically. You do not define them — they
are managed by the library.

## langchain_pg_collection

One row per named collection. This project uses a single collection named `"docs"`.

```sql
CREATE TABLE langchain_pg_collection (
    uuid      UUID  PRIMARY KEY,
    name      TEXT  UNIQUE NOT NULL,  -- "docs"
    cmetadata JSON
);
```

## langchain_pg_embedding

One row per ingested document. This is the main table.

```sql
CREATE TABLE langchain_pg_embedding (
    id            TEXT    PRIMARY KEY,
    collection_id UUID    REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding     vector, -- 1024 dimensions (snowflake-arctic-embed2)
    document      TEXT,   -- chunk body text
    cmetadata     JSONB   -- frontmatter: topic, tags, audience, source, etc.
);

-- GIN index for fast metadata filtering
CREATE INDEX ix_cmetadata_gin ON langchain_pg_embedding
    USING gin (cmetadata jsonb_path_ops);
```

## Inspecting the data

```sql
-- Count ingested docs
SELECT count(*) FROM langchain_pg_embedding;

-- Browse content and metadata
SELECT document, cmetadata->>'topic', cmetadata->>'tags'
FROM langchain_pg_embedding
LIMIT 5;

-- Filter by metadata field
SELECT document FROM langchain_pg_embedding
WHERE cmetadata->>'audience' = 'People Leaders';
```

## Re-ingestion

`ingest/pipeline.py` calls `drop_tables()` then recreates everything before each run.
This ensures re-ingestion is always clean — no stale or duplicate vectors.
The trade-off is that the DB is briefly empty during re-ingest.

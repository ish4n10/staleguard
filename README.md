# StaleGuard

StaleGuard is a post-retrieval audit layer for RAG pipelines.

It sits between retrieval and generation:

```text
Retriever / Vector DB
        ->
Retrieved chunks
        ->
StaleGuard audit
        ->
Decision:
  FRESH
  STALE
  MIXED
  CONFLICTED
  UNKNOWN
        ->
LLM
```

StaleGuard does not replace retrieval. It checks whether retrieved chunks are trustworthy enough to send to the model.

## What It Does

- score chunk freshness from metadata, version, and date signals
- find fresher alternatives from a corpus
- detect contradictions with rules and optional local NLI
- surface schema issues when retrieved metadata is incomplete
- return a provenance object you can use in middleware or UI

Current trust order:

```text
metadata > rules = nli
```

That means:
- explicit metadata wins when it is available
- rules and NLI are secondary evidence sources
- if rules and NLI agree, confidence increases

## Core API

There are two main ways to use the library.

### 1. Audit already-normalized chunks

```python
from src import audit

result = audit(
    query="How do I configure Redis Cluster in Redis 8?",
    retrieved_chunks=[
        {
            "id": "redis_6_cluster_001",
            "text": "Redis 6 uses requirepass configuration for cluster authentication.",
            "product": "redis",
            "topic": "cluster_configuration",
            "version": "6.2",
            "date_ts": 1640995200,
            "source": "redis-6.2-cluster.md",
            "metadata": {"status": "superseded", "superseded_by": "8.0"},
        }
    ],
    corpus=[...],
    use_nli=True,
)

print(result.verdict)
print(result.conflicts)
print(result.fresh_alternatives)
print(result.provenance)
```

### 2. Use the middleware-style entrypoint

```python
from src import StaleGuardConfig, audit_retrieved

config = StaleGuardConfig(
    use_nli=True,
    block_on_conflict=False,
    embedding_model="all-MiniLM-L6-v2",
    conflict_model="cross-encoder/nli-deberta-v3-base",
    similarity_threshold=0.40,
    contradiction_threshold=0.80,
    contradiction_margin=0.15,
    cache_dir=".cache/huggingface",
)

result = audit_retrieved(
    query="How do I configure Redis Cluster in Redis 8?",
    retrieved=retriever_output,
    corpus=corpus,
    config=config,
)
```

`audit_retrieved(...)` auto-detects:

- raw Chroma query results
- LangChain-style documents with `page_content` and `metadata`
- already-normalized chunk dicts

## Verdicts

- `FRESH`: retrieved context looks current
- `STALE`: outdated chunks were found
- `MIXED`: stale and conflicting evidence were found
- `CONFLICTED`: conflicting evidence was found
- `UNKNOWN`: metadata is too incomplete to make a strong judgment

If you want any conflict to block generation, set:

```python
block_on_conflict=True
```

That collapses `MIXED` into `CONFLICTED`.

## Middleware Pattern

This is the intended integration shape:

```python
from src import audit_retrieved

def audited_retrieve(query: str, retriever_output, corpus: list[dict]):
    audit_result = audit_retrieved(
        query=query,
        retrieved=retriever_output,
        corpus=corpus,
        use_nli=True,
        block_on_conflict=False,
    )

    if audit_result.verdict == "CONFLICTED":
        return {"block": True, "audit": audit_result}

    return {"block": False, "audit": audit_result}
```

The application can then:

- send `FRESH` chunks to the LLM
- replace or warn on `STALE`
- block or escalate on `CONFLICTED`
- show provenance on `MIXED`

## Supported Retrieval Inputs

### Chroma

```python
from src import audit_chroma_result

result = audit_chroma_result(
    query=query,
    chroma_result=chroma_result,
    corpus=corpus,
    use_nli=True,
)
```

### LangChain-style documents

```python
from src import audit_langchain_docs

result = audit_langchain_docs(
    query=query,
    docs=docs,
    corpus=corpus,
    use_nli=True,
)
```

The repo also exposes adapter helpers:

- `normalize_chroma_result(...)`
- `normalize_langchain_docs(...)`
- `normalize_chunks(...)`

## Chunk Schema

Best case input:

```python
{
    "id": "redis_8_cluster_001",
    "text": "...",
    "product": "redis",
    "topic": "cluster_configuration",
    "version": "8.0",
    "date_ts": 1735689600,
    "source": "redis-8.0-cluster.md",
    "metadata": {
        "status": "current"
    }
}
```

Audit-critical fields:

- `text`
- `product`
- `topic`
- `version`
- `date_ts`

If some metadata is missing, StaleGuard will:

- infer a few safe fields from `source`
- record `schema_issues`
- lower confidence or return `UNKNOWN` when needed

## Demos

### Redis demo

```bash
python -m src.chroma_demo
```

This builds a local Chroma collection and shows:

- raw Chroma retrieval
- normalized chunks
- prepared chunks
- audit result

### Large engineering demo

```bash
python -m src.engineering_demo
```

This uses a larger corpus around:

- loop engineering
- context engineering
- tool loop policy
- memory policy
- planning
- retrieval policy

The example intentionally includes:

- stale 2024/2025 chunks
- fresher 2026 replacements
- conflicting guidance that triggers NLI

## Current Status

The repo is at the local/offline MVP stage.

## License

[MIT](LICENSE)

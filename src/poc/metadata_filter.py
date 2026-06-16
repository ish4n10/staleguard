import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
ST_CACHE = CACHE_ROOT / "sentence_transformers"
TF_CACHE = CACHE_ROOT / "transformers"
HF_HUB_CACHE = CACHE_ROOT / "hub"

for cache_dir in (CACHE_ROOT, ST_CACHE, TF_CACHE, HF_HUB_CACHE):
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(TF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ST_CACHE)

EMBED_MODEL = "all-MiniLM-L6-v2"

# setup
client = chromadb.PersistentClient(path="./poc3_chroma")
embedder = SentenceTransformer(EMBED_MODEL, cache_folder=str(ST_CACHE))

collection = client.get_or_create_collection("staleguard_poc3")

# ingest chunks with rich metadata
docs = [
    {"id": "redis-v6-auth", "topic": "redis-authentication",
     "version": "6.0", "date_ts": 1609459200,  # 2021-01-01
     "text": "Redis 6: use requirepass in redis.conf for basic auth"},
    {"id": "redis-v7-auth", "topic": "redis-authentication",
     "version": "7.0", "date_ts": 1672531200,  # 2023-01-01
     "text": "Redis 7: use ACL system for fine-grained auth control"},
    {"id": "redis-v8-auth", "topic": "redis-authentication",
     "version": "8.0", "date_ts": 1741046400,  # 2025-03-04
     "text": "Redis 8: ACL v2 with improved permission inheritance"},
    {"id": "fastapi-v068",  "topic": "fastapi-middleware",
     "version": "0.68", "date_ts": 1625097600,  # 2021-07-01
     "text": "FastAPI 0.68: add middleware using app.add_middleware()"},
    {"id": "fastapi-v0115", "topic": "fastapi-middleware",
     "version": "0.115", "date_ts": 1730419200, # 2024-11-01
     "text": "FastAPI 0.115: middleware now supports async lifespan events"},
]

# add to ChromaDB
embeddings = embedder.encode([d["text"] for d in docs]).tolist()
collection.upsert(
    ids=[d["id"] for d in docs],
    embeddings=embeddings,
    documents=[d["text"] for d in docs],
    metadatas=[{"topic": d["topic"], "version": d["version"],
                "date_ts": d["date_ts"]} for d in docs]
)

def find_newer_in_kb(topic: str, chunk_date_ts: int) -> dict | None:
    topic_embedding = embedder.encode([topic]).tolist()
    results = collection.query(
        query_embeddings=topic_embedding,
        n_results=10,
        where={"$and": [
            {"topic": {"$eq": topic}},
            {"date_ts": {"$gt": chunk_date_ts}}   # strictly newer
        ]}
    )
    if not results["ids"][0]:
        return None
    # return newest
    meta_list = results["metadatas"][0]
    doc_list  = results["documents"][0]
    newest_idx = max(range(len(meta_list)),
                     key=lambda i: meta_list[i]["date_ts"])
    return {"id": results["ids"][0][newest_idx],
            "text": doc_list[newest_idx],
            "metadata": meta_list[newest_idx]}

print("=== PoC 3: ChromaDB Metadata Filter ===\n")

# simulate: retrieved chunk is redis-v6-auth (old)
stale_chunk = docs[0]
print(f"Stale chunk: {stale_chunk['id']} (date_ts={stale_chunk['date_ts']})")
newer = find_newer_in_kb(stale_chunk["topic"], stale_chunk["date_ts"])
if newer:
    print(f"Newer found: {newer['id']} v{newer['metadata']['version']}")
    print(f"  Text: {newer['text']}")
else:
    print("No newer chunk found")

print()

# simulate: retrieved chunk is fastapi-v0115 (newest) — should find nothing
latest = docs[4]
print(f"Latest chunk: {latest['id']} (date_ts={latest['date_ts']})")
newer2 = find_newer_in_kb(latest["topic"], latest["date_ts"])
print(f"Newer found: {newer2}")  # should be None

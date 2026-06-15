
import os
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
SENTENCE_TRANSFORMERS_CACHE = MODEL_CACHE_ROOT / "sentence_transformers"
TRANSFORMERS_CACHE = MODEL_CACHE_ROOT / "transformers"
HUGGINGFACE_HUB_CACHE = MODEL_CACHE_ROOT / "hub"

for cache_dir in (
    MODEL_CACHE_ROOT,
    SENTENCE_TRANSFORMERS_CACHE,
    TRANSFORMERS_CACHE,
    HUGGINGFACE_HUB_CACHE,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUGGINGFACE_HUB_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_CACHE)

# loads ~180MB once, cached locally after first download
model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base",
    cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
)
label_mapping = ['contradiction', 'entailment', 'neutral']

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def should_check_pair(text_a: str, text_b: str, min_similarity: float = 0.4) -> bool:
    embs = embedder.encode([text_a, text_b], normalize_embeddings=True)
    similarity = float(np.dot(embs[0], embs[1]))
    return similarity >= min_similarity


def detect_conflict(chunk_a: str, chunk_b: str) -> dict:
    """
    Returns conflict analysis between two text chunks.
    contradiction_score > 0.7 → flag as conflict.
    """
    scores = model.predict([(chunk_a, chunk_b)], apply_softmax=True)
    probs = scores[0]  # shape: [3] — contradiction, entailment, neutral
    
    contradiction = probs[0]
    entailment = probs[1]
    neutral = probs[2]

    is_conflict = (
        contradiction > 0.8
        and contradiction == max(probs)
        and contradiction - max(entailment, neutral) > 0.15 
    )
    label = label_mapping[np.argmax(probs)]
    
    return {
        "contradiction_score": round(float(probs[0]), 3),
        "entailment_score":    round(float(probs[1]), 3),
        "neutral_score":       round(float(probs[2]), 3),
        "label":               label,
        "is_conflict":         is_conflict
    }

def check_chunk_set(chunks: list[dict]) -> list[dict]:
    """
    Given N retrieved chunks, check all pairs for conflicts.
    Returns list of conflicting pairs only.
    O(n²) — fine for n <= 10 retrieved chunks.
    """
    conflicts = []
    
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if should_check_pair(chunks[i]["text"], chunks[j]["text"]):
                result = detect_conflict(chunks[i]["text"], chunks[j]["text"])
                
                if result["is_conflict"]:
                    conflicts.append({
                        "chunk_a_id":           chunks[i]["id"],
                        "chunk_b_id":           chunks[j]["id"],
                        "chunk_a_text":         chunks[i]["text"][:80],
                        "chunk_b_text":         chunks[j]["text"][:80],
                        "contradiction_score":  result["contradiction_score"],
                    })
    
    return conflicts


# ── test cases ────────────────────────────────────────────────────────────────

print("=== StaleGuard PoC 2: Conflict Detector ===\n")

# simulate 4 chunks that your RAG returned for a query about Redis limits
retrieved_chunks = [
    {
        "id": "redis-v6-limits",
        "text": "Redis 6 allows a maximum of 10,000 concurrent connections by default."
    },
    {
        "id": "redis-v8-limits", 
        "text": "Redis 8 removed the default connection limit entirely — connections are now bounded only by system resources."
    },
    {
        "id": "redis-v6-auth",
        "text": "Redis 6 authentication uses a single password via the requirepass directive."
    },
    {
        "id": "redis-v8-auth",
        "text": "Redis 8 authentication requires ACL-based user management. The requirepass directive is deprecated."
    },
    {
        "id" : "redis_mem",
        "text": "Redis uses memory"
    },
    {
        "id": "redis_ram",
        "text": "redis keeps data in ram"
    }
]

conflicts = check_chunk_set(retrieved_chunks)

if conflicts:
    print(f"⚠️  Found {len(conflicts)} conflict(s) in retrieved chunks:\n")
    for c in conflicts:
        print(f"  CONFLICT: {c['chunk_a_id']} vs {c['chunk_b_id']}")
        print(f"  Score: {c['contradiction_score']}")
        print(f"  A: {c['chunk_a_text']}...")
        print(f"  B: {c['chunk_b_text']}...")
        print()
else:
    print("✓ No conflicts detected")


# ── also test some non-conflicting pairs so you can see the difference ────────

print("--- Non-conflict sanity check ---\n")

safe_chunks = [
    {"id": "doc1", "text": "Redis is an in-memory data structure store."},
    {"id": "doc2", "text": "Redis stores data in RAM for fast access."},
    {"id": "doc3", "text": "Redis supports strings, hashes, lists, and sets."},
]

safe_conflicts = check_chunk_set(safe_chunks)
print(f"Non-conflicting chunks → conflicts found: {len(safe_conflicts)} (should be 0)")


from sentence_transformers import CrossEncoder
import numpy as np

# loads ~180MB once, cached locally after first download
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
label_mapping = ['contradiction', 'entailment', 'neutral']

def detect_conflict(chunk_a: str, chunk_b: str) -> dict:
    """
    Returns conflict analysis between two text chunks.
    contradiction_score > 0.7 → flag as conflict.
    """
    scores = model.predict([(chunk_a, chunk_b)])
    probs = scores[0]  # shape: [3] — contradiction, entailment, neutral
    
    label = label_mapping[np.argmax(probs)]
    
    return {
        "contradiction_score": round(float(probs[0]), 3),
        "entailment_score":    round(float(probs[1]), 3),
        "neutral_score":       round(float(probs[2]), 3),
        "label":               label,
        "is_conflict":         float(probs[0]) > 0.7
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
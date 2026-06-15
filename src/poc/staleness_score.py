import datetime


# freshness_score = e ** (-decay_rate * age_in_mins) 

import math 
from datetime import datetime

def freshness_score(age_months: float, decay_rate: float) -> float:
    return math.exp(-decay_rate * age_months)

def parse_date(date_str: str) -> datetime | None:
    formats = ["%Y-%m-%d", "%B %Y", "%b %Y", "%Y-%m", "%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def age_in_months(doc_date: datetime) -> float:
    return (datetime.now() - doc_date).days / 30.44

def freshness_score(age_months: float, decay_rate: float = 0.05) -> float:
    if age_months < 0:
        return 1.0   # future date = brand new
    return round(math.exp(-decay_rate * age_months), 3)

# ── fake knowledge base ───────────────────────────────

KB = {
    "redis-configuration": [
        {"id": "redis-v6", "date": "2022-01-15",
         "text": "Use maxmemory-policy allkeys-lru in Redis 6"},
        {"id": "redis-v8", "date": "2025-03-10",
         "text": "Redis 8 ACL v2 with improved permission inheritance"},
    ],
    "fastapi-routing": [
        {"id": "fastapi-068", "date": "2021-07-01",
         "text": "Use @app.get decorator for FastAPI routes"},
        {"id": "fastapi-115", "date": "2024-11-01",
         "text": "Use APIRouter with lifespan for FastAPI 0.115"},
    ],
    "docker-networking": [
        {"id": "docker-2024", "date": "2027-06-01",
         "text": "Use bridge networks for container communication"},
        # only one chunk — nothing is newer than this
    ]
}

def find_newer(topic: str, chunk_date: datetime) -> dict | None:
    chunks = KB.get(topic, [])
    newer = [c for c in chunks if parse_date(c["date"]) > chunk_date]
    return max(newer, key=lambda c: parse_date(c["date"])) if newer else None

# ── staleness scorer ──────────────────────────────────

def score_chunk(chunk: dict, topic: str) -> dict:
    date = parse_date(chunk["date"])
    
    if date is None:
        return {
            "id": chunk["id"],
            "freshness_score": 0.5,
            "verdict": "UNKNOWN",
            "reason": "no date metadata found"
        }
    
    age    = age_in_months(date)
    if (age < 0):
        return {
            "id": chunk["id"],
            "freshness_score": 0,
            "verdict": "Invalid (Future Data)",
            "reason": "no date metadata found"
        }
    score  = freshness_score(age)
    newer  = find_newer(topic, date)
    
    if newer:
        verdict = "STALE"
        reason  = f"superseded by {newer['id']} ({newer['date']})"
    elif score < 0.4:
        verdict = "AGING"
        reason  = f"no newer chunk found but {age:.0f} months old"
    else:
        verdict = "FRESH"
        reason  = "most recent chunk on this topic"
    
    return {
        "id":              chunk["id"],
        "date":            chunk["date"],
        "age_months":      round(age, 1),
        "freshness_score": score,
        "verdict":         verdict,
        "reason":          reason,
        "newer_available": newer["id"] if newer else None
    }

# ── simulate RAG retrieving old chunks ───────────────

print("=== StaleGuard PoC 1: Staleness Scorer ===\n")

# these are the chunks your RAG returned (the old ones)
retrieved = [
    (KB["redis-configuration"][0],  "redis-configuration"),
    (KB["fastapi-routing"][0],       "fastapi-routing"),
    (KB["docker-networking"][0],     "docker-networking"),
]

for chunk, topic in retrieved:
    result = score_chunk(chunk, topic)
    print(f"{'⚠️  STALE' if result['verdict']=='STALE' else '✓  FRESH'} | {result['id']}")
    if 'date' in result : print(f"   Date: {result['date']}  |  Age: {result['age_months']} months")
    print(f"   Freshness score: {result['freshness_score']}")
    print(f"   Verdict: {result['verdict']}  —  {result['reason']}")
    if 'newer_available' in result:
        print(f"   → Use instead: {result['newer_available']}")
    print()
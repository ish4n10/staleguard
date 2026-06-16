import os
import re
from pathlib import Path

import numpy as np

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

from sentence_transformers import CrossEncoder, SentenceTransformer

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL = "all-MiniLM-L6-v2"
LABELS = ["contradiction", "entailment", "neutral"]

MIN_SIMILARITY = 0.45
MIN_SEMANTIC_ONLY_SIMILARITY = 0.85
MIN_VERSION_CONFLICT_SIMILARITY = 0.65
MIN_CONTRADICTION_SCORE = 0.8
MIN_CONTRADICTION_MARGIN = 0.15
MIN_ENTAILMENT_SCORE = 0.75

VERSION_PATTERNS = (
    re.compile(r"\b(?:version|v)\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Za-z0-9._-]*\s+(\d+(?:\.\d+)*)\b"),
    re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE),
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "uses", "using", "via", "with",
}

_cross_encoder: CrossEncoder | None = None
_embedder: SentenceTransformer | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(NLI_MODEL, cache_folder=str(ST_CACHE))
    return _cross_encoder


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, cache_folder=str(ST_CACHE))
    return _embedder


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }


def extract_versions(text: str, chunk_id: str = "") -> set[str]:
    haystack = f"{text} {chunk_id}"
    return {
        match.group(1)
        for pattern in VERSION_PATTERNS
        for match in pattern.finditer(haystack)
    }


def common_tokens(chunks: list[dict]) -> set[str]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        for token in content_tokens(chunk["text"]):
            counts[token] = counts.get(token, 0) + 1

    threshold = max(2, len(chunks) // 2 + 1)
    return {token for token, count in counts.items() if count >= threshold}


def similarity(text_a: str, text_b: str) -> float:
    embedder = get_embedder()
    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def predict_probabilities(pairs: list[tuple[str, str]]) -> np.ndarray:
    model = get_cross_encoder()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(pairs)
    return model.predict(pairs, apply_softmax=True)


def score_pair(text_a: str, text_b: str) -> dict:
    forward = predict_probabilities([(text_a, text_b)])[0]
    reverse = predict_probabilities([(text_b, text_a)])[0]
    probs = (forward + reverse) / 2

    contradiction = float(probs[0])
    entailment = float(probs[1])
    neutral = float(probs[2])
    label = LABELS[int(np.argmax(probs))]

    return {
        "contradiction_score": round(contradiction, 3),
        "entailment_score": round(entailment, 3),
        "neutral_score": round(neutral, 3),
        "label": label,
        "is_direct_contradiction": (
            label == "contradiction"
            and contradiction >= MIN_CONTRADICTION_SCORE
            and contradiction - max(entailment, neutral) >= MIN_CONTRADICTION_MARGIN
        ),
    }


def analyze_pair(chunk_a: dict, chunk_b: dict, shared_corpus_tokens: set[str]) -> dict:
    text_a = chunk_a["text"]
    text_b = chunk_b["text"]
    pair_similarity = similarity(text_a, text_b)
    versions_a = extract_versions(text_a, chunk_a["id"])
    versions_b = extract_versions(text_b, chunk_b["id"])
    anchor_tokens = sorted((content_tokens(text_a) & content_tokens(text_b)) - shared_corpus_tokens)

    base = {
        "chunk_a_id": chunk_a["id"],
        "chunk_b_id": chunk_b["id"],
        "similarity": round(pair_similarity, 3),
        "shared_anchor_tokens": anchor_tokens,
        "versions_a": sorted(versions_a),
        "versions_b": sorted(versions_b),
    }

    if pair_similarity < MIN_SIMILARITY:
        return {
            **base,
            "checked": False,
            "reason": "low_similarity",
            "classification": "unrelated",
        }

    if not anchor_tokens and pair_similarity < MIN_SEMANTIC_ONLY_SIMILARITY:
        return {
            **base,
            "checked": False,
            "reason": "no_anchor_overlap",
            "classification": "unrelated",
        }

    nli = score_pair(text_a, text_b)
    version_mismatch = bool(versions_a and versions_b and versions_a != versions_b)

    classification = "unrelated"
    if nli["is_direct_contradiction"]:
        classification = "direct_contradiction"
    elif nli["label"] == "entailment" and nli["entailment_score"] >= MIN_ENTAILMENT_SCORE:
        classification = "same_fact"
    elif anchor_tokens and version_mismatch and pair_similarity >= MIN_VERSION_CONFLICT_SIMILARITY:
        classification = "possible_version_conflict"

    return {
        **base,
        **nli,
        "checked": True,
        "reason": "scored",
        "classification": classification,
        "is_conflict": classification in {"direct_contradiction", "possible_version_conflict"},
    }


def analyze_chunk_set(chunks: list[dict]) -> list[dict]:
    shared_corpus_tokens = common_tokens(chunks)
    return [
        analyze_pair(chunks[i], chunks[j], shared_corpus_tokens)
        for i in range(len(chunks))
        for j in range(i + 1, len(chunks))
    ]


def check_chunk_set(chunks: list[dict]) -> list[dict]:
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    conflicts = []

    for analysis in analyze_chunk_set(chunks):
        if not analysis.get("is_conflict"):
            continue

        chunk_a = chunk_by_id[analysis["chunk_a_id"]]
        chunk_b = chunk_by_id[analysis["chunk_b_id"]]
        conflicts.append(
            {
                "chunk_a_id": analysis["chunk_a_id"],
                "chunk_b_id": analysis["chunk_b_id"],
                "classification": analysis["classification"],
                "chunk_a_text": chunk_a["text"][:80],
                "chunk_b_text": chunk_b["text"][:80],
                "contradiction_score": analysis["contradiction_score"],
            }
        )

    return conflicts


if __name__ == "__main__":
    retrieved_chunks = [
        {"id": "redis-v6-limits", "text": "Redis 6 allows a maximum of 10,000 concurrent connections by default."},
        {"id": "redis-v8-limits", "text": "Redis 8 removed the default connection limit entirely - connections are now bounded only by system resources."},
        {"id": "redis-v6-auth", "text": "Redis 6 authentication uses a single password via the requirepass directive."},
        {"id": "redis-v8-auth", "text": "Redis 8 authentication requires ACL-based user management. The requirepass directive is deprecated."},
        {"id": "redis_mem", "text": "Redis uses memory."},
        {"id": "redis_ram", "text": "Redis keeps data in RAM."},
    ]

    analyses = analyze_chunk_set(retrieved_chunks)
    for analysis in analyses:
        pair_name = f"{analysis['chunk_a_id']} vs {analysis['chunk_b_id']}"
        if not analysis["checked"]:
            print(f"{pair_name}: skipped ({analysis['reason']})")
            continue

        print(
            f"{pair_name}: {analysis['classification']} ({analysis['label']}) "
            f"[c={analysis['contradiction_score']}, e={analysis['entailment_score']}, "
            f"n={analysis['neutral_score']}, sim={analysis['similarity']}]"
        )

    conflicts = [analysis for analysis in analyses if analysis.get("is_conflict")]
    print()
    print(conflicts if conflicts else "No conflicts detected.")

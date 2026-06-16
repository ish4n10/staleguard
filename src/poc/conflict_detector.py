import os
import re
from pathlib import Path

import numpy as np

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
os.environ["HF_HUB_CACHE"] = str(HUGGINGFACE_HUB_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUGGINGFACE_HUB_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_CACHE)

from sentence_transformers import CrossEncoder, SentenceTransformer

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LABEL_MAPPING = ["contradiction", "entailment", "neutral"]
MIN_SIMILARITY = 0.45
MIN_SEMANTIC_ONLY_SIMILARITY = 0.85
MIN_CONTRADICTION_SCORE = 0.8
MIN_CONTRADICTION_MARGIN = 0.15
MIN_ENTAILMENT_SCORE = 0.75
MIN_VERSION_CONFLICT_SIMILARITY = 0.65
VERSION_PATTERNS = (
    re.compile(r"\b(?:version|v)\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Za-z0-9._-]*\s+(\d+(?:\.\d+)*)\b"),
)
ID_VERSION_PATTERN = re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "uses",
    "using",
    "via",
    "with",
}

_cross_encoder: CrossEncoder | None = None
_embedder: SentenceTransformer | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(
            NLI_MODEL_NAME,
            cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
        )
    return _cross_encoder


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=str(SENTENCE_TRANSFORMERS_CACHE),
        )
    return _embedder


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _content_tokens(text: str) -> set[str]:
    tokens = _tokenize(text)
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }


def _extract_versions(text: str, chunk_id: str = "") -> set[str]:
    versions = set()

    for pattern in VERSION_PATTERNS:
        versions.update(match.group(1) for match in pattern.finditer(text))

    if chunk_id:
        versions.update(match.group(1) for match in ID_VERSION_PATTERN.finditer(chunk_id))

    return versions


def _build_corpus_common_tokens(chunks: list[dict]) -> set[str]:
    doc_frequency: dict[str, int] = {}

    for chunk in chunks:
        for token in _content_tokens(chunk["text"]):
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    threshold = max(2, len(chunks) // 2 + 1)
    return {token for token, count in doc_frequency.items() if count >= threshold}


def _pair_similarity(text_a: str, text_b: str) -> float:
    embedder = _get_embedder()
    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def predict_probabilities(pairs: list[tuple[str, str]]) -> np.ndarray:
    model = _get_cross_encoder()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(pairs)
    return model.predict(pairs, apply_softmax=True)


def detect_conflict(chunk_a: str, chunk_b: str) -> dict:
    forward_probs = predict_probabilities([(chunk_a, chunk_b)])[0]
    reverse_probs = predict_probabilities([(chunk_b, chunk_a)])[0]
    probs = (forward_probs + reverse_probs) / 2

    contradiction = float(probs[0])
    entailment = float(probs[1])
    neutral = float(probs[2])
    strongest_non_contradiction = max(entailment, neutral)
    label = LABEL_MAPPING[int(np.argmax(probs))]

    return {
        "contradiction_score": round(contradiction, 3),
        "entailment_score": round(entailment, 3),
        "neutral_score": round(neutral, 3),
        "label": label,
        "is_direct_contradiction": (
            label == "contradiction"
            and contradiction >= MIN_CONTRADICTION_SCORE
            and contradiction - strongest_non_contradiction >= MIN_CONTRADICTION_MARGIN
        ),
    }


def analyze_pair(chunk_a: dict, chunk_b: dict, corpus_common_tokens: set[str]) -> dict:
    text_a = chunk_a["text"]
    text_b = chunk_b["text"]
    tokens_a = _content_tokens(text_a)
    tokens_b = _content_tokens(text_b)
    shared_anchor_tokens = sorted((tokens_a & tokens_b) - corpus_common_tokens)
    versions_a = _extract_versions(text_a, chunk_a["id"])
    versions_b = _extract_versions(text_b, chunk_b["id"])

    similarity = _pair_similarity(text_a, text_b)
    if similarity < MIN_SIMILARITY:
        return {
            "chunk_a_id": chunk_a["id"],
            "chunk_b_id": chunk_b["id"],
            "checked": False,
            "reason": "low_similarity",
            "classification": "unrelated",
            "similarity": round(similarity, 3),
            "shared_anchor_tokens": shared_anchor_tokens,
            "versions_a": sorted(versions_a),
            "versions_b": sorted(versions_b),
        }

    if not shared_anchor_tokens and similarity < MIN_SEMANTIC_ONLY_SIMILARITY:
        return {
            "chunk_a_id": chunk_a["id"],
            "chunk_b_id": chunk_b["id"],
            "checked": False,
            "reason": "no_anchor_overlap",
            "classification": "unrelated",
            "similarity": round(similarity, 3),
            "shared_anchor_tokens": shared_anchor_tokens,
            "versions_a": sorted(versions_a),
            "versions_b": sorted(versions_b),
        }

    result = detect_conflict(text_a, text_b)
    version_mismatch = bool(versions_a and versions_b and versions_a != versions_b)

    classification = "unrelated"
    if result["is_direct_contradiction"]:
        classification = "direct_contradiction"
    elif (
        result["label"] == "entailment"
        and result["entailment_score"] >= MIN_ENTAILMENT_SCORE
    ):
        classification = "same_fact"
    elif (
        shared_anchor_tokens
        and version_mismatch
        and similarity >= MIN_VERSION_CONFLICT_SIMILARITY
    ):
        classification = "possible_version_conflict"

    return {
        "chunk_a_id": chunk_a["id"],
        "chunk_b_id": chunk_b["id"],
        "checked": True,
        "reason": "scored",
        "similarity": round(similarity, 3),
        "shared_anchor_tokens": shared_anchor_tokens,
        "versions_a": sorted(versions_a),
        "versions_b": sorted(versions_b),
        "classification": classification,
        "is_conflict": classification in {"direct_contradiction", "possible_version_conflict"},
        **result,
    }


def analyze_chunk_set(chunks: list[dict]) -> list[dict]:
    analyses = []
    corpus_common_tokens = _build_corpus_common_tokens(chunks)

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            analyses.append(analyze_pair(chunks[i], chunks[j], corpus_common_tokens))

    return analyses


def check_chunk_set(chunks: list[dict]) -> list[dict]:
    conflicts = []
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}

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
        {
            "id": "redis-v6-limits",
            "text": "Redis 6 allows a maximum of 10,000 concurrent connections by default.",
        },
        {
            "id": "redis-v8-limits",
            "text": "Redis 8 removed the default connection limit entirely - connections are now bounded only by system resources.",
        },
        {
            "id": "redis-v6-auth",
            "text": "Redis 6 authentication uses a single password via the requirepass directive.",
        },
        {
            "id": "redis-v8-auth",
            "text": "Redis 8 authentication requires ACL-based user management. The requirepass directive is deprecated.",
        },
        {
            "id": "redis_mem",
            "text": "Redis uses memory.",
        },
        {
            "id": "redis_ram",
            "text": "Redis keeps data in RAM.",
        },
    ]

    analyses = analyze_chunk_set(retrieved_chunks)
    for analysis in analyses:
        pair_name = f"{analysis['chunk_a_id']} vs {analysis['chunk_b_id']}"
        if not analysis["checked"]:
            detail = analysis.get("similarity", analysis["reason"])
            print(f"{pair_name}: skipped ({detail})")
            continue

        print(
            f"{pair_name}: {analysis['classification']} ({analysis['label']}) "
            f"[c={analysis['contradiction_score']}, e={analysis['entailment_score']}, "
            f"n={analysis['neutral_score']}, sim={analysis['similarity']}]"
        )

    conflicts = [analysis for analysis in analyses if analysis.get("is_conflict")]
    print()
    print(conflicts if conflicts else "No conflicts detected.")

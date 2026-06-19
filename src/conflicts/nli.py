import os
from pathlib import Path
from typing import Any

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

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LABELS = ["contradiction", "entailment", "neutral"]
MIN_SIMILARITY = 0.40
MIN_CONTRADICTION_SCORE = 0.8
MIN_CONTRADICTION_MARGIN = 0.15

_model = None
_embedder = None


def get_nli_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, cache_folder=str(ST_CACHE))
    return _model


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=str(ST_CACHE))
    return _embedder


def pair_similarity(text_a: str, text_b: str) -> float:
    embedder = get_embedder()
    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def predict_probabilities(text_a: str, text_b: str) -> np.ndarray:
    model = get_nli_model()

    if hasattr(model, "predict_proba"):
        forward = model.predict_proba([(text_a, text_b)])[0]
        reverse = model.predict_proba([(text_b, text_a)])[0]
    else:
        forward = model.predict([(text_a, text_b)], apply_softmax=True)[0]
        reverse = model.predict([(text_b, text_a)], apply_softmax=True)[0]

    return (forward + reverse) / 2


def score_chunk_pair(chunk_a: dict[str, Any], chunk_b: dict[str, Any]) -> dict[str, Any]:
    similarity = pair_similarity(chunk_a["text"], chunk_b["text"])
    if similarity < MIN_SIMILARITY:
        return {
            "chunk_a": chunk_a["id"],
            "chunk_b": chunk_b["id"],
            "type": "NLI_CONFLICT",
            "confidence": 0.0,
            "label": "skipped",
            "similarity": round(similarity, 3),
            "scores": None,
            "is_conflict": False,
            "reason": f"similarity below threshold ({MIN_SIMILARITY:.2f})",
        }

    probs = predict_probabilities(chunk_a["text"], chunk_b["text"])

    contradiction = float(probs[0])
    entailment = float(probs[1])
    neutral = float(probs[2])
    label = LABELS[int(np.argmax(probs))]
    strongest_other = max(entailment, neutral)

    return {
        "chunk_a": chunk_a["id"],
        "chunk_b": chunk_b["id"],
        "type": "NLI_CONFLICT",
        "confidence": round(contradiction, 3),
        "label": label,
        "similarity": round(similarity, 3),
        "scores": {
            "contradiction": round(contradiction, 3),
            "entailment": round(entailment, 3),
            "neutral": round(neutral, 3),
        },
        "is_conflict": (
            label == "contradiction"
            and contradiction >= MIN_CONTRADICTION_SCORE
            and contradiction - strongest_other >= MIN_CONTRADICTION_MARGIN
        ),
    }

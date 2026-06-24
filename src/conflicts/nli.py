from typing import Any

import numpy as np

from ..config import StaleGuardConfig
from ..providers import get_conflict_provider, get_embedding_provider

LABELS = ["contradiction", "entailment", "neutral"]


def pair_similarity(text_a: str, text_b: str, config: StaleGuardConfig) -> float:
    provider = get_embedding_provider(config)
    return provider.similarity(text_a, text_b)


def predict_probabilities(text_a: str, text_b: str, config: StaleGuardConfig) -> np.ndarray:
    provider = get_conflict_provider(config)
    return np.asarray(provider.predict_probabilities(text_a, text_b), dtype=float)


def score_chunk_pair(
    chunk_a: dict[str, Any],
    chunk_b: dict[str, Any],
    config: StaleGuardConfig | None = None,
) -> dict[str, Any]:
    config = config or StaleGuardConfig()
    similarity = pair_similarity(chunk_a["text"], chunk_b["text"], config)
    if similarity < config.similarity_threshold:
        return {
            "chunk_a": chunk_a["id"],
            "chunk_b": chunk_b["id"],
            "source": "nli",
            "type": "NLI_CONFLICT",
            "confidence": 0.0,
            "label": "skipped",
            "similarity": round(similarity, 3),
            "scores": None,
            "is_conflict": False,
            "reason": f"similarity below threshold ({config.similarity_threshold:.2f})",
        }

    probs = predict_probabilities(chunk_a["text"], chunk_b["text"], config)

    contradiction = float(probs[0])
    entailment = float(probs[1])
    neutral = float(probs[2])
    label = LABELS[int(np.argmax(probs))]
    strongest_other = max(entailment, neutral)

    return {
        "chunk_a": chunk_a["id"],
        "chunk_b": chunk_b["id"],
        "source": "nli",
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
            and contradiction >= config.contradiction_threshold
            and contradiction - strongest_other >= config.contradiction_margin
        ),
    }

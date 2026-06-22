import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import StaleGuardConfig

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LABELS = ["contradiction", "entailment", "neutral"]

_model_cache: dict[tuple[str, str], Any] = {}
_embedder_cache: dict[tuple[str, str], Any] = {}


def _cache_paths(config: StaleGuardConfig) -> dict[str, Path]:
    cache_root = Path(config.cache_dir) if config.cache_dir else Path(".cache") / "huggingface"
    return {
        "root": cache_root,
        "sentence_transformers": cache_root / "sentence_transformers",
        "transformers": cache_root / "transformers",
        "hub": cache_root / "hub",
    }


def _configure_cache_env(config: StaleGuardConfig) -> dict[str, Path]:
    paths = _cache_paths(config)
    for cache_dir in paths.values():
        cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(paths["root"])
    os.environ["HF_HUB_CACHE"] = str(paths["hub"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths["hub"])
    os.environ["TRANSFORMERS_CACHE"] = str(paths["transformers"])
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(paths["sentence_transformers"])
    return paths


def local_model_path(model_name: str, config: StaleGuardConfig) -> str:
    st_cache = _configure_cache_env(config)["sentence_transformers"]
    model_dir = st_cache / f"models--{model_name.replace('/', '--')}"
    ref_file = model_dir / "refs" / "main"
    if not ref_file.exists():
        return model_name

    snapshot_id = ref_file.read_text(encoding="utf-8").strip()
    snapshot_dir = model_dir / "snapshots" / snapshot_id
    if not snapshot_dir.exists():
        return model_name

    return str(snapshot_dir)


def get_nli_model(config: StaleGuardConfig):
    if config.conflict_backend != "local":
        raise ValueError(f"Unsupported conflict backend: {config.conflict_backend}")

    cache_key = (config.conflict_backend, config.conflict_model)
    model = _model_cache.get(cache_key)
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(local_model_path(config.conflict_model, config))
        _model_cache[cache_key] = model
    return model


def get_embedder(config: StaleGuardConfig):
    if config.embedding_backend != "local":
        raise ValueError(f"Unsupported embedding backend: {config.embedding_backend}")

    cache_key = (config.embedding_backend, config.embedding_model)
    embedder = _embedder_cache.get(cache_key)
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(local_model_path(config.embedding_model, config))
        _embedder_cache[cache_key] = embedder
    return embedder


def pair_similarity(text_a: str, text_b: str, config: StaleGuardConfig) -> float:
    embedder = get_embedder(config)
    embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def predict_probabilities(text_a: str, text_b: str, config: StaleGuardConfig) -> np.ndarray:
    model = get_nli_model(config)

    if hasattr(model, "predict_proba"):
        forward = model.predict_proba([(text_a, text_b)])[0]
        reverse = model.predict_proba([(text_b, text_a)])[0]
    else:
        forward = model.predict([(text_a, text_b)], apply_softmax=True)[0]
        reverse = model.predict([(text_b, text_a)], apply_softmax=True)[0]

    return (forward + reverse) / 2


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

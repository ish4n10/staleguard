import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .config import StaleGuardConfig

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class EmbeddingProvider(Protocol):
    def similarity(self, text_a: str, text_b: str) -> float:
        ...


class ConflictProvider(Protocol):
    def predict_probabilities(self, text_a: str, text_b: str) -> Sequence[float]:
        ...


EmbeddingProviderFactory = Callable[[StaleGuardConfig], EmbeddingProvider]
ConflictProviderFactory = Callable[[StaleGuardConfig], ConflictProvider]

_embedding_factories: dict[str, EmbeddingProviderFactory] = {}
_conflict_factories: dict[str, ConflictProviderFactory] = {}
_model_cache: dict[tuple[str, str], Any] = {}
_embedder_cache: dict[tuple[str, str], Any] = {}


def register_embedding_provider(name: str, factory: EmbeddingProviderFactory) -> None:
    _embedding_factories[name] = factory


def register_conflict_provider(name: str, factory: ConflictProviderFactory) -> None:
    _conflict_factories[name] = factory


def get_embedding_provider(config: StaleGuardConfig) -> EmbeddingProvider:
    if config.embedding_provider is not None:
        return cast(EmbeddingProvider, config.embedding_provider)

    factory = _embedding_factories.get(config.embedding_backend)
    if factory is None:
        raise ValueError(f"Unsupported embedding backend: {config.embedding_backend}")
    return factory(config)


def get_conflict_provider(config: StaleGuardConfig) -> ConflictProvider:
    if config.conflict_provider is not None:
        return cast(ConflictProvider, config.conflict_provider)

    factory = _conflict_factories.get(config.conflict_backend)
    if factory is None:
        raise ValueError(f"Unsupported conflict backend: {config.conflict_backend}")
    return factory(config)


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


def _local_model_path(model_name: str, config: StaleGuardConfig) -> str:
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


class LocalSentenceTransformerEmbeddingProvider:
    def __init__(self, config: StaleGuardConfig) -> None:
        self.config = config

    def _get_embedder(self) -> Any:
        cache_key = (self.config.embedding_backend, self.config.embedding_model)
        embedder = _embedder_cache.get(cache_key)
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(_local_model_path(self.config.embedding_model, self.config))
            _embedder_cache[cache_key] = embedder
        return embedder

    def similarity(self, text_a: str, text_b: str) -> float:
        embedder = self._get_embedder()
        embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))


class LocalCrossEncoderConflictProvider:
    def __init__(self, config: StaleGuardConfig) -> None:
        self.config = config

    def _get_model(self) -> Any:
        cache_key = (self.config.conflict_backend, self.config.conflict_model)
        model = _model_cache.get(cache_key)
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(_local_model_path(self.config.conflict_model, self.config))
            _model_cache[cache_key] = model
        return model

    def predict_probabilities(self, text_a: str, text_b: str) -> Sequence[float]:
        model = self._get_model()
        if hasattr(model, "predict_proba"):
            forward = model.predict_proba([(text_a, text_b)])[0]
            reverse = model.predict_proba([(text_b, text_a)])[0]
        else:
            forward = model.predict([(text_a, text_b)], apply_softmax=True)[0]
            reverse = model.predict([(text_b, text_a)], apply_softmax=True)[0]

        return ((forward + reverse) / 2).tolist()


register_embedding_provider("local", LocalSentenceTransformerEmbeddingProvider)
register_conflict_provider("local", LocalCrossEncoderConflictProvider)

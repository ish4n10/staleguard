from dataclasses import dataclass
from pathlib import Path


@dataclass
class StaleGuardConfig:
    use_nli: bool = False
    block_on_conflict: bool = False

    embedding_backend: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_base: str | None = None

    conflict_backend: str = "local"
    conflict_model: str = "cross-encoder/nli-deberta-v3-base"
    conflict_api_base: str | None = None

    similarity_threshold: float = 0.40
    contradiction_threshold: float = 0.80
    contradiction_margin: float = 0.15

    cache_dir: str | Path | None = None

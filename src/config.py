from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    embedding_provider: Any = field(default=None, repr=False, compare=False)
    conflict_provider: Any = field(default=None, repr=False, compare=False)

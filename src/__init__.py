__version__ = "0.1.0"

from .api import StaleGuard
from .auditor import audit, audit_chroma_result, audit_langchain_docs, audit_retrieved
from .config import StaleGuardConfig
from .audit_types import AuditResult
from .providers import (
    ConflictProvider,
    EmbeddingProvider,
    register_conflict_provider,
    register_embedding_provider,
)


def evaluate_case(*args, **kwargs):
    from .eval import evaluate_case as _evaluate_case

    return _evaluate_case(*args, **kwargs)


def run_eval(*args, **kwargs):
    from .eval import run_eval as _run_eval

    return _run_eval(*args, **kwargs)

__all__ = [
    "audit",
    "audit_chroma_result",
    "audit_langchain_docs",
    "audit_retrieved",
    "__version__",
    "StaleGuard",
    "StaleGuardConfig",
    "AuditResult",
    "evaluate_case",
    "run_eval",
    "EmbeddingProvider",
    "ConflictProvider",
    "register_embedding_provider",
    "register_conflict_provider",
]

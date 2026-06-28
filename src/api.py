from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .audit_types import AuditResult
from .auditor import audit as audit_chunks_impl
from .auditor import audit_retrieved
from .config import StaleGuardConfig


CONFIG_FIELD_NAMES = {field.name for field in fields(StaleGuardConfig)}


def _clean_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in overrides.items()
        if key in CONFIG_FIELD_NAMES and value is not None
    }


class StaleGuard:
    def __init__(
        self,
        config: StaleGuardConfig | None = None,
        **overrides: Any,
    ) -> None:
        cleaned = _clean_overrides(overrides)
        self.config = replace(config, **cleaned) if config is not None else StaleGuardConfig(**cleaned)

    @classmethod
    def from_backends(
        cls,
        *,
        embedding_backend: str = "local",
        conflict_backend: str = "local",
        **overrides: Any,
    ) -> "StaleGuard":
        return cls(
            embedding_backend=embedding_backend,
            conflict_backend=conflict_backend,
            **overrides,
        )

    @classmethod
    def from_providers(
        cls,
        *,
        embedding_provider: Any = None,
        conflict_provider: Any = None,
        **overrides: Any,
    ) -> "StaleGuard":
        return cls(
            embedding_provider=embedding_provider,
            conflict_provider=conflict_provider,
            **overrides,
        )

    def audit(
        self,
        query: str,
        retrieved: object,
        corpus: list[dict] | None = None,
    ) -> AuditResult:
        return audit_retrieved(
            query=query,
            retrieved=retrieved,
            corpus=corpus,
            config=self.config,
        )

    def audit_chunks(
        self,
        query: str,
        retrieved_chunks: list[dict],
        corpus: list[dict] | None = None,
    ) -> AuditResult:
        return audit_chunks_impl(
            query=query,
            retrieved_chunks=retrieved_chunks,
            corpus=corpus,
            config=self.config,
        )

    def evaluate_case(
        self,
        case: dict,
        corpus: list[dict],
        *,
        use_nli: bool | None = None,
        block_on_conflict: bool | None = None,
    ) -> dict:
        from .eval import evaluate_case

        return evaluate_case(
            case=case,
            corpus=corpus,
            config=self._config_with_runtime_flags(use_nli, block_on_conflict),
        )

    def run_eval(
        self,
        corpus_path: str | Path,
        cases_path: str | Path,
        *,
        use_nli: bool | None = None,
        block_on_conflict: bool | None = None,
    ) -> dict:
        from .eval import run_eval

        return run_eval(
            corpus_path=corpus_path,
            cases_path=cases_path,
            config=self._config_with_runtime_flags(use_nli, block_on_conflict),
        )

    def with_config(self, **overrides: Any) -> "StaleGuard":
        return StaleGuard(self.config, **overrides)

    def _config_with_runtime_flags(
        self,
        use_nli: bool | None,
        block_on_conflict: bool | None,
    ) -> StaleGuardConfig:
        overrides = _clean_overrides(
            {
                "use_nli": self.config.use_nli if use_nli is None else use_nli,
                "block_on_conflict": (
                    self.config.block_on_conflict
                    if block_on_conflict is None
                    else block_on_conflict
                ),
            }
        )
        return replace(self.config, **overrides)

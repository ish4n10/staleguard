from dataclasses import dataclass

@dataclass
class AuditResult:
    verdict: str
    confidence: float
    stale_chunks: list
    conflicts: list
    fresh_alternatives: list
    provenance: dict


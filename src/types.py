from dataclasses import dataclass

@dataclass
class AuditResult:
    verdict: str
    confidence: float
    stable_chunks: list
    conflicts: list
    fresh_alts: list
    provenance: dict


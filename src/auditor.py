from .alternatives.finder import find_fresh_alternatives
from .conflicts.rules import detect_rule_conflicts
from .scorers.freshness import score_chunk_freshness
from .types import AuditResult


def build_provenance_card(
    verdict: str,
    confidence: float,
    retrieved_chunks: list[dict],
    stale_chunks: list[dict],
    fresh_alternatives: list[dict],
    conflicts: list[dict],
) -> dict:
    recommendation = _recommendation(stale_chunks, fresh_alternatives, conflicts)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "total_chunks_checked": len(retrieved_chunks),
        "stale_count": len(stale_chunks),
        "conflict_count": len(conflicts),
        "used_chunks": [chunk.get("id") for chunk in retrieved_chunks],
        "stale_chunks": stale_chunks,
        "fresh_alternatives": fresh_alternatives,
        "conflicts": conflicts,
        "recommendation": recommendation,
    }


def audit(
    query: str,
    retrieved_chunks: list[dict],
    corpus: list[dict] | None = None,
) -> AuditResult:
    stale_chunks = [
        result
        for result in (score_chunk_freshness(chunk, query, corpus) for chunk in retrieved_chunks)
        if result["verdict"] in {"STALE", "AGING"}
    ]
    fresh_alternatives = find_fresh_alternatives(retrieved_chunks, corpus, query)
    conflicts = detect_rule_conflicts(retrieved_chunks)

    verdict, confidence = _verdict_and_confidence(retrieved_chunks, stale_chunks, conflicts)
    provenance = build_provenance_card(
        verdict=verdict,
        confidence=confidence,
        retrieved_chunks=retrieved_chunks,
        stale_chunks=stale_chunks,
        fresh_alternatives=fresh_alternatives,
        conflicts=conflicts,
    )

    return AuditResult(
        verdict=verdict,
        confidence=confidence,
        stale_chunks=stale_chunks,
        conflicts=conflicts,
        fresh_alternatives=fresh_alternatives,
        provenance=provenance,
    )


def _recommendation(
    stale_chunks: list[dict],
    fresh_alternatives: list[dict],
    conflicts: list[dict],
) -> str:
    if conflicts:
        return "Block generation or ask user to resolve conflicting chunks."
    if stale_chunks and fresh_alternatives:
        return "Regenerate answer using fresher chunks."
    if stale_chunks:
        return "Warn user that retrieved context may be outdated."
    return "Proceed with retrieved chunks."


def _verdict_and_confidence(
    retrieved_chunks: list[dict],
    stale_chunks: list[dict],
    conflicts: list[dict],
) -> tuple[str, float]:
    if conflicts and stale_chunks:
        return "MIXED", 0.75
    if conflicts:
        return "CONFLICTED", 0.8
    if stale_chunks:
        return "STALE", 0.7
    if retrieved_chunks:
        return "FRESH", 0.85
    return "UNKNOWN", 0.0

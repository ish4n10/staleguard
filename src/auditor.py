from .types import AuditResult
from scorers.freshness import score_chunk_freshness
from alternatives.finder import find_fresh_alternatives
from conflicts.rules import detect_rule_conflicts


def build_provenance_card(
    verdict: str,
    confidence: float,
    retrieved_chunks: list[dict],
    stale_chunks: list[dict],
    fresh_alternatives: list[dict],
    conflicts: list[dict],
) -> dict:
    if conflicts:
        recommendation = "Block generation or ask user to resolve conflicting chunks."
    elif stale_chunks and fresh_alternatives:
        recommendation = "Regenerate answer using fresher chunks."
    elif stale_chunks:
        recommendation = "Warn user that retrieved context may be outdated."
    else:
        recommendation = "Proceed with retrieved chunks."

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

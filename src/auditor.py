from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from scorers.freshness import score_chunk_freshness
from alternatives.finder import find_fresh_alternatives
from conflicts.rules import detect_rule_conflicts

_TYPES_PATH = Path(__file__).with_name("types.py")
_TYPES_SPEC = spec_from_file_location("staleguard_local_types", _TYPES_PATH)
_TYPES_MODULE = module_from_spec(_TYPES_SPEC)
assert _TYPES_SPEC.loader is not None
_TYPES_SPEC.loader.exec_module(_TYPES_MODULE)
AuditResult = _TYPES_MODULE.AuditResult


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


def audit(
    query: str,
    retrieved_chunks: list[dict],
    corpus: list[dict] | None = None,
) -> AuditResult:
    stale_chunks = [] 
    for chunk in retrieved_chunks: 
        result = score_chunk_freshness(chunk, query, corpus)
        if result['verdict'] in ("STALE", "AGING"):
            stale_chunks.append(result)

        
    fresh_alternatives = find_fresh_alternatives(
        retrieved_chunks, corpus, query
    )

    conflicts = detect_rule_conflicts(retrieved_chunks)


    if conflicts and stale_chunks: 
        verdict = 'MIXED'
        confidence = 0.75
    elif conflicts:
        verdict = "CONFLICTED"
        confidence = 0.8
    elif stale_chunks:
        verdict = 'STALE'
        confidence = 0.7
    elif retrieved_chunks:
        verdict = "FRESH"
        confidence = 0.85
    else: 
        verdict = "UNKNOWN"
        confidence = 0.0

    provenance = {
        "verdict": verdict,
        "confidence": confidence,
        "total_chunks_checked": len(retrieved_chunks),
        "stale_count": len(stale_chunks),
        "conflict_count": len(conflicts),
        "used_chunks": [chunk["id"] for chunk in retrieved_chunks],
        "stale_chunks": stale_chunks,
        "fresh_alternatives": fresh_alternatives,
        "conflicts": conflicts,
    }

    return AuditResult(
        verdict=verdict,
        confidence=confidence,
        stale_chunks=stale_chunks,
        conflicts=conflicts,
        fresh_alternatives=fresh_alternatives,
        provenance=provenance,
    )

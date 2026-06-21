from .adapters.chroma import normalize_chroma_result
from .alternatives.finder import find_fresh_alternatives
from .audit_types import AuditResult
from .conflicts.rules import detect_rule_conflicts
from .scorers.freshness import score_chunk_freshness
from .conflicts.nli import score_chunk_pair
from .schema import prepare_chunks
from itertools import combinations


def build_provenance_card(
    verdict: str,
    confidence: float,
    retrieved_chunks: list[dict],
    stale_chunks: list[dict],
    fresh_alternatives: list[dict],
    conflicts: list[dict],
) -> dict:
    recommendation = _recommendation(stale_chunks, fresh_alternatives, conflicts)
    schema_issues = collect_schema_issues(retrieved_chunks)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "total_chunks_checked": len(retrieved_chunks),
        "stale_count": len(stale_chunks),
        "conflict_count": len(conflicts),
        "used_chunks": [chunk.get("id") for chunk in retrieved_chunks],
        "schema_summary": {
            "chunks_with_schema_issues": len(schema_issues),
            "chunks_with_inferred_fields": sum(1 for issue in schema_issues if issue["inferred_fields"]),
            "chunks_missing_required_fields": sum(
                1 for issue in schema_issues if issue["missing_required_fields"]
            ),
            "chunks_missing_audit_fields": sum(
                1 for issue in schema_issues if issue["missing_audit_fields"]
            ),
        },
        "schema_issues": schema_issues,
        "stale_chunks": stale_chunks,
        "fresh_alternatives": fresh_alternatives,
        "conflicts": conflicts,
        "recommendation": recommendation,
    }


def collect_schema_issues(chunks: list[dict]) -> list[dict]:
    issues = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        inferred_fields = metadata.get("staleguard_inferred_fields", [])
        missing_required = metadata.get("staleguard_missing_required", [])
        missing_audit = metadata.get("staleguard_missing_audit_fields", [])

        if not inferred_fields and not missing_required and not missing_audit:
            continue

        issues.append(
            {
                "id": chunk.get("id"),
                "source": chunk.get("source"),
                "inferred_fields": inferred_fields,
                "missing_required_fields": missing_required,
                "missing_audit_fields": missing_audit,
            }
        )

    return issues

def collect_nli_conflicts(chunks: list[dict]) -> list[dict]:
    conflicts = [] 
    
    for chunk_a, chunk_b in combinations(chunks, 2):
        if chunk_a.get('product') != chunk_b.get('product'):
            continue
        if chunk_a.get('topic') != chunk_b.get('topic'):
            continue
        if chunk_a.get('version') == chunk_b.get('version'):
            continue

        result = score_chunk_pair(chunk_a, chunk_b)
        if result['is_conflict']:
            conflicts.append(result)

    return conflicts
        
def merge_conflicts(rule_conflicts, nli_conflicts) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}

    for conflict in rule_conflicts:
        pair = tuple(sorted((conflict["chunk_a"], conflict["chunk_b"])))
        merged[pair] = {
            "chunk_a": pair[0],
            "chunk_b": pair[1],
            "type": conflict["type"],
            "source": "rules",
            "sources": ["rules"],
            "reason": conflict.get("reason"),
            "confidence": conflict.get("confidence", 0.75),
            "rule_confidence": conflict.get("confidence", 0.75),
            "nli_confidence": None,
            "shared_terms": conflict.get("shared_terms", []),
        }

    for conflict in nli_conflicts:
        pair = tuple(sorted((conflict["chunk_a"], conflict["chunk_b"])))
        entry = merged.get(pair)
        nli_confidence = conflict.get("confidence", 0.0)
        if entry is None:
            merged[pair] = {
                "chunk_a": pair[0],
                "chunk_b": pair[1],
                "type": conflict["type"],
                "source": "nli",
                "sources": ["nli"],
                "reason": conflict.get("reason"),
                "confidence": nli_confidence,
                "rule_confidence": None,
                "nli_confidence": nli_confidence,
                "label": conflict.get("label"),
                "similarity": conflict.get("similarity"),
                "scores": conflict.get("scores"),
            }
            continue

        if "nli" not in entry["sources"]:
            entry["sources"].append("nli")
        entry["source"] = "rules+nli"
        entry["nli_confidence"] = nli_confidence
        entry["label"] = conflict.get("label")
        entry["similarity"] = conflict.get("similarity")
        entry["scores"] = conflict.get("scores")
        entry["confidence"] = min(0.98, max(entry["rule_confidence"] or 0.0, nli_confidence) + 0.1)
        entry["reason"] = "Rules and NLI both indicate a version conflict"

    return list(merged.values())


def audit(
    query: str,
    retrieved_chunks: list[dict],
    corpus: list[dict] | None = None,
    use_nli: bool = False
) -> AuditResult:
    prepared_retrieved = prepare_chunks(retrieved_chunks)
    prepared_corpus = prepare_chunks(corpus)

    stale_chunks = [
        result
        for result in (
            score_chunk_freshness(chunk, query, prepared_corpus)
            for chunk in prepared_retrieved
        )
        if result["verdict"] in {"STALE", "AGING"}
    ]
    fresh_alternatives = find_fresh_alternatives(prepared_retrieved, prepared_corpus, query)
    rule_conflicts = detect_rule_conflicts(prepared_retrieved)


    if use_nli:
        nli_conflicts = collect_nli_conflicts(prepared_retrieved)
        conflicts = merge_conflicts(rule_conflicts, nli_conflicts)

    else: 
        conflicts = rule_conflicts
        
    verdict, confidence = _verdict_and_confidence(prepared_retrieved, stale_chunks, conflicts)
    provenance = build_provenance_card(
        verdict=verdict,
        confidence=confidence,
        retrieved_chunks=prepared_retrieved,
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


def audit_chroma_result(
    query: str,
    chroma_result: dict,
    corpus: list[dict] | None = None,
    use_nli: bool = False,
) -> AuditResult:
    retrieved_chunks = normalize_chroma_result(chroma_result)
    return audit(
        query=query,
        retrieved_chunks=retrieved_chunks,
        corpus=corpus,
        use_nli=use_nli,
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
    if any(chunk.get("metadata", {}).get("staleguard_missing_required") for chunk in retrieved_chunks):
        return "UNKNOWN", 0.0
    if conflicts and stale_chunks:
        confidence = min(0.98, max(_stale_confidence(stale_chunks), _conflict_confidence(conflicts)) + 0.05)
        return "MIXED", confidence
    if conflicts:
        return "CONFLICTED", _conflict_confidence(conflicts)
    if stale_chunks:
        return "STALE", _stale_confidence(stale_chunks)
    if retrieved_chunks and any(
        chunk.get("metadata", {}).get("staleguard_missing_audit_fields")
        for chunk in retrieved_chunks
    ):
        return "UNKNOWN", 0.4
    if retrieved_chunks:
        return "FRESH", 0.85
    return "UNKNOWN", 0.0


def _stale_confidence(stale_chunks: list[dict]) -> float:
    if not stale_chunks:
        return 0.0
    return max(chunk.get("confidence", 0.7) or 0.7 for chunk in stale_chunks)


def _conflict_confidence(conflicts: list[dict]) -> float:
    if not conflicts:
        return 0.0
    return max(conflict.get("confidence", 0.75) or 0.75 for conflict in conflicts)

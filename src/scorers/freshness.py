import math
import re
from datetime import datetime
from typing import Any

SECONDS_PER_MONTH = 30 * 24 * 60 * 60
VERSION_PATTERN = re.compile(r"\b(\d+(?:\.\d+)*)\b")
Chunk = dict[str, Any]


def temporal_score(doc_ts: int, now_ts: int, decay_rate: float = 0.05) -> float:
    age_months = (now_ts - doc_ts) / SECONDS_PER_MONTH
    return math.exp(-decay_rate * age_months)


def parse_version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None

    match = VERSION_PATTERN.search(str(value))
    if not match:
        return None

    return tuple(int(part) for part in match.group(1).split("."))


def version_is_requested(query: str, chunk: Chunk) -> bool:
    query_version = parse_version(query)
    chunk_version = parse_version(chunk.get("version"))
    return bool(query_version and chunk_version and query_version == chunk_version)


def is_newer_version(candidate: Chunk, chunk: Chunk) -> bool:
    candidate_version = parse_version(candidate.get("version"))
    chunk_version = parse_version(chunk.get("version"))
    if candidate_version and chunk_version:
        return candidate_version > chunk_version
    return candidate.get("date_ts", 0) > chunk.get("date_ts", 0)


def find_superseding_chunk(chunk: Chunk, corpus: list[Chunk] | None) -> Chunk | None:
    if not corpus:
        return None

    candidates = [
        candidate
        for candidate in corpus
        if candidate.get("id") != chunk.get("id")
        and candidate.get("product") == chunk.get("product")
        and candidate.get("topic") == chunk.get("topic")
        and is_newer_version(candidate, chunk)
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: (
            parse_version(candidate.get("version")) or (),
            candidate.get("date_ts", 0),
        ),
    )


def score_chunk_freshness(
    chunk: Chunk,
    query: str,
    corpus: list[Chunk] | None = None,
    now_ts: int | None = None,
    decay_rate: float = 0.05,
) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    chunk_id = chunk.get("id")
    chunk_date_ts = chunk.get("date_ts")

    if now_ts is None:
        now_ts = int(datetime.now().timestamp())

    if version_is_requested(query, chunk):
        return _result(
            chunk_id,
            1.0,
            "FRESH",
            f"user explicitly requested version {chunk.get('version')}",
            source="query",
            confidence=1.0,
        )

    superseded_by = metadata.get("superseded_by")
    if metadata.get("status") == "superseded" or superseded_by:
        return _result(
            chunk_id,
            0.1,
            "STALE",
            f"metadata marks chunk as superseded by {superseded_by or 'a newer version'}",
            superseded_by,
            source="metadata",
            confidence=0.95,
        )

    if not chunk_date_ts:
        return _result(
            chunk_id,
            0.5,
            "UNKNOWN",
            "missing date_ts metadata",
            source="metadata_missing",
            confidence=0.4,
        )

    freshness = round(temporal_score(chunk_date_ts, now_ts, decay_rate), 3)
    newer_chunk = find_superseding_chunk(chunk, corpus)

    if newer_chunk:
        return _result(
            chunk_id,
            freshness,
            "STALE",
            f"superseded by newer chunk {newer_chunk['id']}",
            newer_chunk["id"],
            source="corpus_comparison",
            confidence=0.8,
        )

    if freshness < 0.4:
        return _result(
            chunk_id,
            freshness,
            "AGING",
            "no fresher replacement found, but chunk is old",
            source="temporal",
            confidence=0.6,
        )

    return _result(
        chunk_id,
        freshness,
        "FRESH",
        "chunk appears current for its topic",
        source="temporal",
        confidence=max(0.5, freshness),
    )


def _result(
    chunk_id: str | None,
    freshness_score: float,
    verdict: str,
    reason: str,
    newer_available: str | None = None,
    source: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "freshness_score": freshness_score,
        "verdict": verdict,
        "reason": reason,
        "newer_available": newer_available,
        "source": source,
        "confidence": confidence,
    }

import math
import re
from datetime import datetime

SECONDS_PER_MONTH = 30 * 24 * 60 * 60
VERSION_PATTERN = re.compile(r"\b(\d+(?:\.\d+)*)\b")


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


def version_is_requested(query: str, chunk: dict) -> bool:
    query_version = parse_version(query)
    chunk_version = parse_version(chunk.get("version"))
    return bool(query_version and chunk_version and query_version == chunk_version)


def is_newer_version(candidate: dict, chunk: dict) -> bool:
    candidate_version = parse_version(candidate.get("version"))
    chunk_version = parse_version(chunk.get("version"))
    if candidate_version and chunk_version:
        return candidate_version > chunk_version
    return candidate.get("date_ts", 0) > chunk.get("date_ts", 0)


def find_superseding_chunk(chunk: dict, corpus: list[dict] | None) -> dict | None:
    if not corpus:
        return None

    matches = [
        candidate
        for candidate in corpus
        if candidate.get("id") != chunk.get("id")
        and candidate.get("product") == chunk.get("product")
        and candidate.get("topic") == chunk.get("topic")
        and is_newer_version(candidate, chunk)
    ]
    if not matches:
        return None

    return max(
        matches,
        key=lambda candidate: (
            parse_version(candidate.get("version")) or (),
            candidate.get("date_ts", 0),
        ),
    )


def score_chunk_freshness(
    chunk: dict,
    query: str,
    corpus: list[dict] | None = None,
    now_ts: int | None = None,
    decay_rate: float = 0.05,
) -> dict:
    metadata = chunk.get("metadata", {})
    chunk_id = chunk.get("id")
    chunk_date_ts = chunk.get("date_ts")

    if now_ts is None:
        now_ts = int(datetime.now().timestamp())

    if version_is_requested(query, chunk):
        return {
            "id": chunk_id,
            "freshness_score": 1.0,
            "verdict": "FRESH",
            "reason": f"user explicitly requested version {chunk.get('version')}",
            "newer_available": None,
        }

    superseded_by = metadata.get("superseded_by")
    if metadata.get("status") == "superseded" or superseded_by:
        return {
            "id": chunk_id,
            "freshness_score": 0.1,
            "verdict": "STALE",
            "reason": f"metadata marks chunk as superseded by {superseded_by or 'a newer version'}",
            "newer_available": superseded_by,
        }

    if not chunk_date_ts:
        return {
            "id": chunk_id,
            "freshness_score": 0.5,
            "verdict": "UNKNOWN",
            "reason": "missing date_ts metadata",
            "newer_available": None,
        }

    freshness = round(temporal_score(chunk_date_ts, now_ts, decay_rate), 3)
    newer_chunk = find_superseding_chunk(chunk, corpus)

    if newer_chunk:
        return {
            "id": chunk_id,
            "freshness_score": freshness,
            "verdict": "STALE",
            "reason": f"superseded by newer chunk {newer_chunk['id']}",
            "newer_available": newer_chunk["id"],
        }

    if freshness < 0.4:
        return {
            "id": chunk_id,
            "freshness_score": freshness,
            "verdict": "AGING",
            "reason": "no fresher replacement found, but chunk is old",
            "newer_available": None,
        }

    return {
        "id": chunk_id,
        "freshness_score": freshness,
        "verdict": "FRESH",
        "reason": "chunk appears current for its topic",
        "newer_available": None,
    }

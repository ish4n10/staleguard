from scorers.freshness import find_superseding_chunk, score_chunk_freshness


def find_fresh_alternatives(
    retrieved_chunks: list[dict],
    corpus: list[dict] | None,
    query: str,
    now_ts: int | None = None,
) -> list[dict]:
    if not corpus:
        return []

    alternatives = []

    for chunk in retrieved_chunks:
        freshness = score_chunk_freshness(
            chunk=chunk,
            query=query,
            corpus=corpus,
            now_ts=now_ts,
        )
        if freshness["verdict"] != "STALE":
            continue

        replacement = find_superseding_chunk(chunk, corpus)
        if not replacement:
            continue

        alternatives.append(
            {
                "instead_of": chunk["id"],
                "use": replacement["id"],
                "reason": (
                    f"Same product/topic, newer version available: "
                    f"{chunk.get('version')} -> {replacement.get('version')}"
                ),
            }
        )

    return alternatives

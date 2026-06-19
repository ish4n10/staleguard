from collections.abc import Iterable, Mapping
from typing import Any

FIELD_CANDIDATES = {
    "id": ("id", "chunk_id", "document_id", "doc_id"),
    "text": ("text", "page_content", "content", "document"),
    "product": ("product",),
    "topic": ("topic",),
    "version": ("version",),
    "date_ts": ("date_ts", "timestamp", "created_at_ts", "updated_at_ts"),
    "source": ("source", "path", "uri", "filename", "file_name"),
}


def normalize_chunk(
    raw_chunk: Mapping[str, Any],
    index: int = 0,
    field_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    metadata = raw_chunk.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    resolved_map = dict(field_map or {})

    normalized = {
        "id": _resolve_value(raw_chunk, metadata, "id", resolved_map) or f"chunk_{index}",
        "text": _resolve_value(raw_chunk, metadata, "text", resolved_map),
        "product": _resolve_value(raw_chunk, metadata, "product", resolved_map),
        "topic": _resolve_value(raw_chunk, metadata, "topic", resolved_map),
        "version": _resolve_value(raw_chunk, metadata, "version", resolved_map),
        "date_ts": _resolve_value(raw_chunk, metadata, "date_ts", resolved_map),
        "source": _resolve_value(raw_chunk, metadata, "source", resolved_map),
        "metadata": metadata,
    }

    if not normalized["text"]:
        raise ValueError("retrieved chunk is missing text/page_content/content")

    return normalized


def normalize_chunks(
    raw_chunks: Iterable[Mapping[str, Any]],
    field_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        normalize_chunk(raw_chunk, index=index, field_map=field_map)
        for index, raw_chunk in enumerate(raw_chunks)
    ]


def _resolve_value(
    raw_chunk: Mapping[str, Any],
    metadata: Mapping[str, Any],
    field_name: str,
    field_map: Mapping[str, str],
) -> Any:
    explicit_name = field_map.get(field_name)
    if explicit_name:
        return raw_chunk.get(explicit_name, metadata.get(explicit_name))

    for candidate in FIELD_CANDIDATES[field_name]:
        if candidate in raw_chunk:
            return raw_chunk[candidate]
        if candidate in metadata:
            return metadata[candidate]

    return None

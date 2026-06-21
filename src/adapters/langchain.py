from collections.abc import Iterable, Mapping
from typing import Any

from .json_adapter import normalize_chunk


def normalize_langchain_doc(
    doc: Any,
    index: int = 0,
    field_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    raw_chunk = {
        "page_content": getattr(doc, "page_content", None),
        "metadata": getattr(doc, "metadata", None),
    }

    if isinstance(doc, Mapping):
        raw_chunk["page_content"] = doc.get("page_content", raw_chunk["page_content"])
        raw_chunk["metadata"] = doc.get("metadata", raw_chunk["metadata"])
        for key in ("id", "text", "source", "product", "topic", "version", "date_ts"):
            if key in doc:
                raw_chunk[key] = doc[key]

    return normalize_chunk(raw_chunk, index=index, field_map=field_map)


def normalize_langchain_docs(
    docs: Iterable[Any],
    field_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        normalize_langchain_doc(doc, index=index, field_map=field_map)
        for index, doc in enumerate(docs)
    ]

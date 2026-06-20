from collections.abc import Mapping
from typing import Any

from .json_adapter import normalize_chunk


def _unwrap_rows(values: Any) -> list[Any]:
    if values is None:
        return []

    if isinstance(values, list) and values and isinstance(values[0], list):
        return values[0]

    return list(values)


def _value_at(values: list[Any], index: int) -> Any:
    if index >= len(values):
        return None

    return values[index]


def normalize_chroma_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = _unwrap_rows(result.get("ids"))
    documents = _unwrap_rows(result.get("documents"))
    metadatas = _unwrap_rows(result.get("metadatas"))
    distances = _unwrap_rows(result.get("distances"))

    count = max(len(ids), len(documents), len(metadatas), len(distances))
    chunks = []

    for index in range(count):
        current_metadata = dict(_value_at(metadatas, index) or {})
        current_distance = _value_at(distances, index)

        if current_distance is not None:
            current_metadata["chroma_distance"] = current_distance

        raw_chunk = {
            "id": _value_at(ids, index),
            "text": _value_at(documents, index),
            "metadata": current_metadata,
        }

        chunks.append(normalize_chunk(raw_chunk, index=index))

    return chunks



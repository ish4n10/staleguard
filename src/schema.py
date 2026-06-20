import re
from copy import deepcopy
from typing import Any

REQUIRED_FIELDS = ("id", "text")
AUDIT_FIELDS = ("product", "topic", "version", "date_ts")
VERSION_PATTERN = re.compile(r"\b(\d+(?:\.\d+)*)\b")
DATE_PATTERN = re.compile(r"\b(20\d{2})[-_](\d{2})[-_](\d{2})\b")


def prepare_chunks(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not chunks:
        return []

    return [prepare_chunk(chunk, index=index) for index, chunk in enumerate(chunks)]


def prepare_chunk(chunk: dict[str, Any], index: int = 0) -> dict[str, Any]:
    prepared = deepcopy(chunk)
    prepared["id"] = prepared.get("id") or f"chunk_{index}"
    prepared["metadata"] = _metadata_dict(prepared.get("metadata"))

    inferred_fields: dict[str, Any] = {}

    source = prepared.get("source") or prepared["metadata"].get("source")
    text = prepared.get("text", "")

    for field_name in ("product", "topic", "version", "date_ts", "source"):
        if prepared.get(field_name) is None and prepared["metadata"].get(field_name) is not None:
            prepared[field_name] = prepared["metadata"][field_name]

    if prepared.get("source") is None and source is not None:
        prepared["source"] = source

    product = prepared.get("product") or infer_product(source)
    if prepared.get("product") is None and product is not None:
        prepared["product"] = product
        inferred_fields["product"] = product

    version = prepared.get("version") or infer_version(prepared.get("source"), text)
    if prepared.get("version") is None and version is not None:
        prepared["version"] = version
        inferred_fields["version"] = version

    topic = prepared.get("topic") or infer_topic(prepared.get("source"))
    if prepared.get("topic") is None and topic is not None:
        prepared["topic"] = topic
        inferred_fields["topic"] = topic

    date_ts = prepared.get("date_ts") or infer_date_ts(prepared.get("source"))
    if prepared.get("date_ts") is None and date_ts is not None:
        prepared["date_ts"] = date_ts
        inferred_fields["date_ts"] = date_ts

    missing_required = [field for field in REQUIRED_FIELDS if not prepared.get(field)]
    missing_audit = [field for field in AUDIT_FIELDS if prepared.get(field) is None]

    prepared["metadata"]["staleguard_inferred_fields"] = sorted(inferred_fields)
    prepared["metadata"]["staleguard_missing_required"] = missing_required
    prepared["metadata"]["staleguard_missing_audit_fields"] = missing_audit

    return prepared


def infer_product(source: str | None) -> str | None:
    if not source:
        return None

    stem = source_name(source)
    if not stem:
        return None

    parts = [part for part in re.split(r"[-_/ ]+", stem) if part]
    if not parts:
        return None

    return parts[0].lower()


def infer_topic(source: str | None) -> str | None:
    if not source:
        return None

    stem = source_name(source)
    if not stem:
        return None

    stem = re.sub(r"^\w+[-_ ]+\d+(?:\.\d+)*[-_ ]*", "", stem)
    stem = re.sub(r"[-_ ]+", "_", stem.strip("_- "))
    return stem.lower() or None


def infer_version(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = VERSION_PATTERN.search(str(value))
        if match:
            return match.group(1)
    return None


def infer_date_ts(source: str | None) -> int | None:
    if not source:
        return None

    match = DATE_PATTERN.search(source)
    if not match:
        return None

    year, month, day = (int(part) for part in match.groups())
    from datetime import datetime, timezone

    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def source_name(source: str) -> str:
    source = source.rsplit("/", 1)[-1]
    source = source.rsplit("\\", 1)[-1]
    return source.rsplit(".", 1)[0]


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    return dict(metadata) if isinstance(metadata, dict) else {}

import re
from itertools import combinations

NEGATION_TERMS = {
    "remove",
    "removed",
    "removes",
    "deprecate",
    "deprecated",
    "deprecates",
    "replace",
    "replaced",
    "disabled",
    "unsupported",
    "obsolete",
    "nolonger",
}
ACTIVE_TERMS = {
    "available",
    "enabled",
    "included",
    "lets",
    "rely",
    "required",
    "support",
    "supported",
    "use",
    "uses",
    "using",
    "works",
}
REPLACEMENT_TERMS = {
    "field",
    "migrate",
    "migrated",
    "migration",
    "replace",
    "replaced",
    "replacement",
}
LEGACY_INTERFACE_TERMS = {
    "annotation",
    "servicename",
    "serviceport",
    "v1beta1",
}
CURRENT_INTERFACE_TERMS = {
    "defaultbackend",
    "field",
    "ingressclass",
    "ingressclassname",
    "pathtype",
}
BEHAVIOR_CHANGE_GROUPS = (
    ({"all"}, {"none", "no"}),
    ({"unconfined"}, {"runtimedefault"}),
    ({"single", "shared"}, {"named", "separate"}),
)
VERSION_PATTERNS = (
    re.compile(r"\b(?:version|v)\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Za-z0-9._-]*\s+(\d+(?:\.\d+)*)\b"),
    re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE),
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
}


def normalize_text(text: str) -> str:
    return text.lower().replace("no longer", "nolonger")


def content_terms(text: str) -> set[str]:
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }


def raw_terms(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(normalize_text(text)))


def shared_content_terms(text_a: str, text_b: str) -> set[str]:
    return content_terms(text_a) & content_terms(text_b)


def extract_versions(text: str, chunk_id: str = "") -> set[str]:
    haystack = f"{text} {chunk_id}"
    versions = set()

    for pattern in VERSION_PATTERNS:
        for match in pattern.finditer(haystack):
            versions.add(match.group(1))

    return versions


def has_negative(text: str) -> bool:
    return bool(content_terms(text) & NEGATION_TERMS)


def has_positive(text: str) -> bool:
    return bool(content_terms(text) & ACTIVE_TERMS)


def has_replacement(text: str) -> bool:
    return bool(content_terms(text) & REPLACEMENT_TERMS)


def has_interface_migration(text_a: str, text_b: str) -> bool:
    tokens_a = content_terms(text_a)
    tokens_b = content_terms(text_b)
    split_interface_terms = (
        (tokens_a & LEGACY_INTERFACE_TERMS and tokens_b & CURRENT_INTERFACE_TERMS)
        or (tokens_b & LEGACY_INTERFACE_TERMS and tokens_a & CURRENT_INTERFACE_TERMS)
    )
    return bool(split_interface_terms)


def has_behavior_change(text_a: str, text_b: str) -> bool:
    tokens_a = raw_terms(text_a)
    tokens_b = raw_terms(text_b)
    for left_terms, right_terms in BEHAVIOR_CHANGE_GROUPS:
        if (tokens_a & left_terms and tokens_b & right_terms) or (
            tokens_b & left_terms and tokens_a & right_terms
        ):
            return True
    return False


def detect_rule_conflicts(chunks: list[dict]) -> list[dict]:
    conflicts = []
    for chunk_a, chunk_b in combinations(chunks, 2):
        if chunk_a.get("product") != chunk_b.get("product"):
            continue
        if chunk_a.get("topic") != chunk_b.get("topic"):
            continue
        if chunk_a.get("version") == chunk_b.get("version"):
            continue

        shared_terms = shared_content_terms(chunk_a.get("text", ""), chunk_b.get("text", ""))
        if not shared_terms:
            continue

        a_negative = has_negative(chunk_a["text"])
        a_positive = has_positive(chunk_a["text"])
        b_negative = has_negative(chunk_b["text"])
        b_positive = has_positive(chunk_b["text"])

        if not (
            (a_negative and b_positive)
            or (a_positive and b_negative)
            or has_interface_migration(chunk_a["text"], chunk_b["text"])
            or has_behavior_change(chunk_a["text"], chunk_b["text"])
        ):
            continue

        conflicts.append(
            {
                "chunk_a": chunk_a["id"],
                "chunk_b": chunk_b["id"],
                "source": "rules",
                "type": "VERSION_CHANGE",
                "confidence": 0.75,
                "reason": "Chunks describe a versioned behavior or migration change",
                "shared_terms": sorted(shared_terms),
            }
        )

    return conflicts

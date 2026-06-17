import re 

NEGATION_TERMS = {
    "removed", "deprecated", "replaced", "disabled",
    "unsupported", "obsolete", "no_longer"
}

ACTIVE_TERMS = {
    "available", "enabled", "required", "supported",
    "use", "uses", "using"
}
VERSION_PATTERNS = (
    re.compile(r"\b(?:version|v)\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Za-z0-9._-]*\s+(\d+(?:\.\d+)*)\b"),
    re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE),
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "uses", "using", "via", "with",
}

def normalize_text(text) -> str:
    return text.lower().replace("no longer", "no_longer")


def content_terms(text: str) -> set[str]:
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }

def shared_content_terms(text_a, text_b) -> set[str]:
    return content_terms(text_a) & content_terms(text_b)


def extract_version(text, chunk_id = "") -> str[str]:
    haystack = f"{text} {chunk_id}"
    versions = set()

    for pattern in VERSION_PATTERNS:
        matches = pattern.finditer(haystack)

        for match in matches:
            v = match.group(1)
            versions.add(v)

    return versions
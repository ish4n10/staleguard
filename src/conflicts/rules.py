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


def has_negative(text) -> bool:
    return bool(content_terms(text) & NEGATION_TERMS)

def has_positive(text) -> bool:
    return bool(content_terms(text) & ACTIVE_TERMS)


def detect_rules_conflicts(chunks: list[dict]) -> list[dict]:

    conflicts = [] 

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk_a = chunks[i]
            chunk_b = chunks[j]

            if chunk_a.get('product') != chunk_b.get('product'):
                continue
            if chunk_a.get('topic') != chunk_b.get('topic'):
                continue
            if chunk_a.get('version') != chunk_b.get('version'):
                continue

            version_a = extract_version(chunk_a.get('text'), chunk_a.get('id', ''))
            version_b = extract_version(chunk_b.get('text'), chunk_b.get('id', ''))

            if not version_a or not version_b:
                continue
        
            if version_a == version_b: 
                continue

            shared_terms = shared_content_terms(chunk_a.get('text', ''), chunk_b.get('text', ''))

            if len(shared_terms) < 1:
                continue

            a_negative = has_negative(chunk_a['text'])
            a_positive = has_positive(chunk_a['text'])
            b_negative = has_negative(chunk_b['text'])
            b_positive = has_positive(chunk_b['text'])

            if (a_negative and b_positive) or (a_positive and b_negative):
                conflicts.append({
                    'chunk_a': chunk_a['text'],
                    'chunk_b': chunk_b['text'],

                    'type': "VERSION_CONFLICT",
                    'confidence': 0.75,
                    "reason": "Chunks describe the same topic across versions",
                    "shared_terms" : sorted(shared_terms)
                })

    return conflicts
        
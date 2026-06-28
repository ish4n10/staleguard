import re
from typing import Any

from .config import StaleGuardConfig
from .providers import get_embedding_provider

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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}


def label_tokens(value: Any) -> set[str]:
    if value is None:
        return set()

    return {
        token
        for token in TOKEN_PATTERN.findall(str(value).lower())
        if token not in STOPWORDS
    }


def content_tokens(text: str | None) -> set[str]:
    if not text:
        return set()

    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def shared_content_terms(chunk_a: dict, chunk_b: dict) -> set[str]:
    return content_tokens(chunk_a.get("text")) & content_tokens(chunk_b.get("text"))


def chunk_descriptor(chunk: dict) -> str:
    text_terms = sorted(content_tokens(chunk.get("text")))[0:18]
    parts = [
        str(chunk.get("product") or ""),
        str(chunk.get("topic") or ""),
        str(chunk.get("source") or ""),
        " ".join(text_terms),
    ]
    return " | ".join(part for part in parts if part)


def pair_semantic_similarity(
    chunk_a: dict,
    chunk_b: dict,
    config: StaleGuardConfig,
) -> tuple[float | None, str | None]:
    try:
        provider = get_embedding_provider(config)
        similarity = provider.similarity(chunk_descriptor(chunk_a), chunk_descriptor(chunk_b))
        return float(similarity), None
    except ModuleNotFoundError as exc:
        if exc.name == "sentence_transformers":
            return None, "embedding_provider_unavailable"
        raise


def match_chunk_pair(
    chunk_a: dict,
    chunk_b: dict,
    config: StaleGuardConfig | None = None,
    *,
    semantic_fallback: bool = False,
    strict_topic: bool = False,
) -> dict[str, Any]:
    config = config or StaleGuardConfig()

    product_tokens_a = label_tokens(chunk_a.get("product"))
    product_tokens_b = label_tokens(chunk_b.get("product"))
    topic_tokens_a = label_tokens(chunk_a.get("topic"))
    topic_tokens_b = label_tokens(chunk_b.get("topic"))
    source_tokens_a = label_tokens(chunk_a.get("source"))
    source_tokens_b = label_tokens(chunk_b.get("source"))
    shared_terms = shared_content_terms(chunk_a, chunk_b)

    product_overlap = max(
        jaccard(product_tokens_a, product_tokens_b),
        jaccard(source_tokens_a, source_tokens_b),
    )
    topic_overlap = max(
        jaccard(topic_tokens_a, topic_tokens_b),
        jaccard(source_tokens_a, source_tokens_b),
    )

    product_signal = max(
        1.0 if product_tokens_a and product_tokens_a == product_tokens_b else 0.0,
        product_overlap,
        0.6 if not product_tokens_a or not product_tokens_b else 0.0,
    )
    topic_signal = max(
        1.0 if topic_tokens_a and topic_tokens_a == topic_tokens_b else 0.0,
        topic_overlap,
        min(1.0, len(shared_terms) / 4) if shared_terms else 0.0,
        0.45 if not topic_tokens_a or not topic_tokens_b else 0.0,
    )

    semantic_similarity = None
    semantic_error = None
    if semantic_fallback and (
        product_signal < config.candidate_match_threshold
        or topic_signal < config.candidate_match_threshold
    ):
        semantic_similarity, semantic_error = pair_semantic_similarity(chunk_a, chunk_b, config)
        if semantic_similarity is not None:
            product_signal = max(product_signal, semantic_similarity if product_signal >= 0.45 else 0.0)
            topic_signal = max(topic_signal, semantic_similarity)

    combined_score = round((0.45 * product_signal) + (0.55 * topic_signal), 3)
    matched = (
        combined_score >= config.candidate_match_threshold
        and product_signal >= 0.45
        and topic_signal >= config.candidate_match_threshold
    )
    if not matched and not strict_topic:
        matched = (
            combined_score >= config.candidate_match_threshold
            and product_signal >= 0.45
            and len(shared_terms) >= config.min_shared_terms
        )

    if semantic_similarity is not None and semantic_similarity >= config.semantic_match_threshold:
        reason = "semantic_fallback"
    elif topic_tokens_a and topic_tokens_b and topic_tokens_a == topic_tokens_b:
        reason = "topic_exact"
    elif product_tokens_a and product_tokens_a == product_tokens_b:
        reason = "product_exact"
    elif shared_terms:
        reason = "shared_content_terms"
    else:
        reason = "metadata_overlap"

    return {
        "matched": matched,
        "reason": reason,
        "score": combined_score,
        "product_signal": round(product_signal, 3),
        "topic_signal": round(topic_signal, 3),
        "semantic_similarity": (
            None if semantic_similarity is None else round(semantic_similarity, 3)
        ),
        "semantic_error": semantic_error,
        "shared_terms": sorted(shared_terms),
    }

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import audit_retrieved
from .adapters.chroma import normalize_chroma_result
from .chroma_demo import chroma_metadata, query_collection, rebuild_collection
from .schema import prepare_chunks


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "examples" / "engineering_demo" / "corpus_large.json"
CHROMA_PATH = REPO_ROOT / "demo_data" / "engineering_chroma"
COLLECTION_NAME = "engineering_docs"


class DemoDocument:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_summary(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        summary.append(
            {
                "id": chunk.get("id"),
                "source": chunk.get("source"),
                "topic": chunk.get("topic"),
                "version": chunk.get("version"),
                "distance": metadata.get("chroma_distance"),
                "status": metadata.get("status"),
            }
        )
    return summary


def to_langchain_docs(chunks: list[dict[str, Any]]) -> list[DemoDocument]:
    docs = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        metadata["id"] = chunk.get("id")
        metadata["product"] = chunk.get("product")
        metadata["topic"] = chunk.get("topic")
        metadata["version"] = chunk.get("version")
        metadata["date_ts"] = chunk.get("date_ts")
        metadata["source"] = chunk.get("source")
        docs.append(DemoDocument(page_content=chunk["text"], metadata=metadata))
    return docs


def main() -> None:
    query = "How should a 2026 agent loop manage context, memory, and repeated tool use?"
    corpus = load_corpus()
    _, collection = rebuild_collection(
        corpus=corpus,
        persist_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    chroma_result = query_collection(collection, query=query, n_results=7)
    chroma_chunks = normalize_chroma_result(chroma_result)
    prepared_chroma_chunks = prepare_chunks(chroma_chunks)
    langchain_docs = to_langchain_docs(chroma_chunks)

    chroma_audit = audit_retrieved(
        query=query,
        retrieved=chroma_result,
        corpus=corpus,
        use_nli=True,
        block_on_conflict=False,
    )
    langchain_audit = audit_retrieved(
        query=query,
        retrieved=langchain_docs,
        corpus=corpus,
        use_nli=True,
        block_on_conflict=False,
    )

    output = {
        "query": query,
        "corpus_path": str(CORPUS_PATH),
        "chroma_path": str(CHROMA_PATH),
        "collection_name": COLLECTION_NAME,
        "corpus_size": len(corpus),
        "raw_chroma_result": chroma_result,
        "prepared_chroma_chunks": prepared_chroma_chunks,
        "retrieval_summary": retrieval_summary(prepared_chroma_chunks),
        "langchain_doc_count": len(langchain_docs),
        "chroma_audit_result": asdict(chroma_audit),
        "langchain_audit_result": asdict(langchain_audit),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

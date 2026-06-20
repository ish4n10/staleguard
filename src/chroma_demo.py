import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb

if __package__:
    from . import audit_chroma_result
    from .adapters.chroma import normalize_chroma_result
    from .conflicts.nli import get_embedder
    from .schema import prepare_chunks
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import audit_chroma_result
    from src.adapters.chroma import normalize_chroma_result
    from src.conflicts.nli import get_embedder
    from src.schema import prepare_chunks


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "examples" / "redis_demo" / "corpus_large.json"
CHROMA_PATH = REPO_ROOT / ".cache" / "chroma" / "redis_demo"
COLLECTION_NAME = "redis_docs"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata", {}))
    metadata["product"] = chunk.get("product")
    metadata["topic"] = chunk.get("topic")
    metadata["version"] = chunk.get("version")
    metadata["date_ts"] = chunk.get("date_ts")
    metadata["source"] = chunk.get("source")
    return {key: value for key, value in metadata.items() if value is not None}


def rebuild_collection(
    corpus: list[dict[str, Any]],
    persist_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
):
    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.create_collection(name=collection_name)

    documents = [chunk["text"] for chunk in corpus]
    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    metadatas = [chroma_metadata(chunk) for chunk in corpus]
    ids = [chunk["id"] for chunk in corpus]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return client, collection


def query_collection(
    collection,
    query: str,
    n_results: int = 4,
) -> dict[str, Any]:
    embedder = get_embedder()
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def retrieval_summary(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        summary.append(
            {
                "id": chunk.get("id"),
                "source": chunk.get("source"),
                "product": chunk.get("product"),
                "topic": chunk.get("topic"),
                "version": chunk.get("version"),
                "distance": metadata.get("chroma_distance"),
                "inferred_fields": metadata.get("staleguard_inferred_fields", []),
                "missing_audit_fields": metadata.get("staleguard_missing_audit_fields", []),
            }
        )
    return summary


def main() -> None:
    query = "How do I configure Redis cluster authentication in Redis 8?"
    corpus = load_corpus()
    _, collection = rebuild_collection(corpus)
    chroma_result = query_collection(collection, query=query, n_results=5)
    normalized_chunks = normalize_chroma_result(chroma_result)
    prepared_chunks = prepare_chunks(normalized_chunks)
    audit_result = audit_chroma_result(
        query=query,
        chroma_result=chroma_result,
        corpus=corpus,
        use_nli=True,
    )

    output = {
        "query": query,
        "chroma_path": str(CHROMA_PATH),
        "collection_name": COLLECTION_NAME,
        "corpus_size": len(corpus),
        "raw_chroma_result": chroma_result,
        "normalized_chunks": normalized_chunks,
        "prepared_chunks": prepared_chunks,
        "retrieval_summary": retrieval_summary(prepared_chunks),
        "audit_result": asdict(audit_result),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

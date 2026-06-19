import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from . import audit
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import audit


def main() -> None:
    with open("examples/redis_demo/retrieved_mixed.json", "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    with open("examples/redis_demo/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    result = audit(
        query="How do I configure Redis Cluster in Redis 8?",
        retrieved_chunks=retrieved,
        corpus=corpus,
        use_nli=True,
    )

    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

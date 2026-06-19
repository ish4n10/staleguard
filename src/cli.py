import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .auditor import audit


def _load_chunks(path: str | None) -> list[dict] | None:
    if path is None:
        return None

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of chunks.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit retrieved RAG chunks.")
    audit_parser.add_argument("--query", required=True, help="User query to audit against.")
    audit_parser.add_argument("--retrieved", required=True, help="Path to retrieved chunks JSON.")
    audit_parser.add_argument("--corpus", help="Optional path to full corpus JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "audit":
        retrieved_chunks = _load_chunks(args.retrieved) or []
        corpus = _load_chunks(args.corpus)
        result = audit(args.query, retrieved_chunks, corpus)
        print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

import json
from dataclasses import asdict
from pathlib import Path

from .audit_types import AuditResult
from .auditor import audit, audit_retrieved
from .config import StaleGuardConfig


def load_corpus(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of chunks.")
    return data


def load_cases(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of eval cases.")
    return data


def build_corpus_index(corpus: list[dict]) -> dict[str, dict]:
    return {chunk["id"]: chunk for chunk in corpus}


def materialize_retrieved(case: dict, corpus_index: dict[str, dict]) -> object:
    if "retrieved" in case:
        return case["retrieved"]

    missing_ids = [chunk_id for chunk_id in case["retrieved_ids"] if chunk_id not in corpus_index]
    if missing_ids:
        missing = ", ".join(missing_ids)
        raise KeyError(f"Case {case['name']} references unknown chunk ids: {missing}")
    return [corpus_index[chunk_id] for chunk_id in case["retrieved_ids"]]


def conflict_pairs(conflicts: list[dict]) -> list[list[str]]:
    pairs = {
        tuple(sorted((conflict["chunk_a"], conflict["chunk_b"])))
        for conflict in conflicts
    }
    return [list(pair) for pair in sorted(pairs)]


def stale_chunk_ids(result: AuditResult) -> list[str]:
    return sorted(chunk["id"] for chunk in result.stale_chunks)


def alternative_pairs(result: AuditResult) -> list[dict]:
    pairs = [
        {
            "instead_of": alternative["instead_of"],
            "use": alternative["use"],
        }
        for alternative in result.fresh_alternatives
    ]
    return sorted(pairs, key=lambda item: (item["instead_of"], item["use"]))


def expected_alternative_pairs(case: dict) -> list[dict]:
    expected = case.get("expected", {})
    pairs = expected.get("fresh_alternatives", [])
    return sorted(
        [{"instead_of": item["instead_of"], "use": item["use"]} for item in pairs],
        key=lambda item: (item["instead_of"], item["use"]),
    )


def expected_conflict_pairs(case: dict) -> list[list[str]]:
    expected = case.get("expected", {})
    pairs = expected.get("conflict_pairs", [])
    normalized = [sorted(pair) for pair in pairs]
    return sorted(normalized)


def evaluate_case(
    case: dict,
    corpus: list[dict],
    config: StaleGuardConfig | None = None,
    use_nli: bool = False,
    block_on_conflict: bool = False,
) -> dict:
    corpus_index = build_corpus_index(corpus)
    retrieved = materialize_retrieved(case, corpus_index)
    if isinstance(retrieved, list) and all(isinstance(item, dict) and "text" in item for item in retrieved):
        result = audit(
            query=case["query"],
            retrieved_chunks=retrieved,
            corpus=corpus,
            use_nli=use_nli,
            block_on_conflict=block_on_conflict,
            config=config,
        )
    else:
        result = audit_retrieved(
            query=case["query"],
            retrieved=retrieved,
            corpus=corpus,
            use_nli=use_nli,
            block_on_conflict=block_on_conflict,
            config=config,
        )

    expected = case["expected"]
    actual = {
        "verdict": result.verdict,
        "stale_chunk_ids": stale_chunk_ids(result),
        "conflict_pairs": conflict_pairs(result.conflicts),
        "fresh_alternatives": alternative_pairs(result),
    }
    checks = {
        "verdict": actual["verdict"] == expected["verdict"],
        "stale_chunk_ids": actual["stale_chunk_ids"] == sorted(expected.get("stale_chunk_ids", [])),
        "conflict_pairs": actual["conflict_pairs"] == expected_conflict_pairs(case),
        "fresh_alternatives": actual["fresh_alternatives"] == expected_alternative_pairs(case),
    }

    return {
        "name": case["name"],
        "query": case["query"],
        "passed": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "actual": actual,
        "audit_result": asdict(result),
    }


def run_eval(
    corpus_path: str | Path,
    cases_path: str | Path,
    config: StaleGuardConfig | None = None,
    use_nli: bool = False,
    block_on_conflict: bool = False,
) -> dict:
    corpus = load_corpus(corpus_path)
    cases = load_cases(cases_path)
    case_results = [
        evaluate_case(
            case=case,
            corpus=corpus,
            config=config,
            use_nli=use_nli,
            block_on_conflict=block_on_conflict,
        )
        for case in cases
    ]

    passed = sum(1 for item in case_results if item["passed"])
    failed = len(case_results) - passed

    return {
        "summary": {
            "corpus_path": str(corpus_path),
            "cases_path": str(cases_path),
            "total_cases": len(case_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(case_results), 3) if case_results else 0.0,
            "use_nli": config.use_nli if config is not None else use_nli,
            "block_on_conflict": (
                config.block_on_conflict if config is not None else block_on_conflict
            ),
        },
        "cases": case_results,
    }


def compact_eval_report(report: dict) -> dict:
    failed_cases = []
    for case in report.get("cases", []):
        if case.get("passed"):
            continue
        failed_cases.append(
            {
                "name": case["name"],
                "query": case["query"],
                "checks": case["checks"],
                "expected": case["expected"],
                "actual": case["actual"],
            }
        )

    output = {"summary": report["summary"]}
    if failed_cases:
        output["failed_cases"] = failed_cases
    return output


def main() -> None:
    default_base = Path("eval_cases") / "kubernetes"
    report = run_eval(
        corpus_path=default_base / "corpus.json",
        cases_path=default_base / "cases.json",
    )
    print(json.dumps(compact_eval_report(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

auditor_spec = spec_from_file_location("staleguard_auditor", SRC_DIR / "auditor.py")
auditor_module = module_from_spec(auditor_spec)
assert auditor_spec.loader is not None
auditor_spec.loader.exec_module(auditor_module)
audit = auditor_module.audit


class AuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis_v6_chunk = {
            "id": "redis_6_cluster_001",
            "product": "redis",
            "topic": "cluster_configuration",
            "version": "6.2",
            "date_ts": 1640995200,
            "text": "Redis 6 uses requirepass configuration for cluster authentication.",
            "metadata": {},
        }
        self.redis_v8_chunk = {
            "id": "redis_8_cluster_001",
            "product": "redis",
            "topic": "cluster_configuration",
            "version": "8.0",
            "date_ts": 1735689600,
            "text": "Redis 8 deprecated requirepass and removed it from the recommended cluster authentication setup.",
            "metadata": {"status": "current"},
        }

    def test_audit_marks_stale_and_returns_alternative(self) -> None:
        result = audit(
            query="How do I configure Redis Cluster?",
            retrieved_chunks=[self.redis_v6_chunk],
            corpus=[self.redis_v6_chunk, self.redis_v8_chunk],
        )

        print("\nauditor_stale_result:", result)
        self.assertEqual(result.verdict, "STALE")
        self.assertEqual(len(result.stale_chunks), 1)
        self.assertEqual(len(result.fresh_alternatives), 1)
        self.assertEqual(result.fresh_alternatives[0]["use"], "redis_8_cluster_001")

    def test_audit_returns_conflicted_for_conflicting_chunks(self) -> None:
        result = audit(
            query="How do I configure Redis Cluster?",
            retrieved_chunks=[self.redis_v6_chunk, self.redis_v8_chunk],
            corpus=[self.redis_v6_chunk, self.redis_v8_chunk],
        )

        print("\nauditor_conflict_result:", result)
        self.assertIn(result.verdict, {"MIXED", "CONFLICTED"})
        self.assertGreaterEqual(len(result.conflicts), 1)

    def test_audit_returns_unknown_for_empty_chunks(self) -> None:
        result = audit(query="anything", retrieved_chunks=[], corpus=[])

        print("\nauditor_empty_result:", result)
        self.assertEqual(result.verdict, "UNKNOWN")
        self.assertEqual(result.stale_chunks, [])
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.fresh_alternatives, [])


if __name__ == "__main__":
    unittest.main()

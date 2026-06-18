import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflicts.rules import detect_rule_conflicts


class RuleConflictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis_v6_auth = {
            "id": "redis-v6-auth",
            "product": "redis",
            "topic": "authentication",
            "version": "6.0",
            "text": "Redis 6 uses requirepass configuration for authentication.",
        }
        self.redis_v8_auth = {
            "id": "redis-v8-auth",
            "product": "redis",
            "topic": "authentication",
            "version": "8.0",
            "text": "Redis 8 deprecated requirepass and removed it from the recommended authentication setup.",
        }
        self.redis_v8_memory = {
            "id": "redis-v8-memory",
            "product": "redis",
            "topic": "memory",
            "version": "8.0",
            "text": "Redis 8 supports RAM-based storage for fast access.",
        }

    def test_detects_version_conflict_for_opposing_change_language(self) -> None:
        result = detect_rule_conflicts([self.redis_v6_auth, self.redis_v8_auth])

        print("\nrule_conflict_result:", result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "VERSION_CONFLICT")
        self.assertEqual(result[0]["chunk_a"], "redis-v6-auth")
        self.assertEqual(result[0]["chunk_b"], "redis-v8-auth")

    def test_ignores_unrelated_chunks(self) -> None:
        result = detect_rule_conflicts([self.redis_v6_auth, self.redis_v8_memory])

        print("\nunrelated_conflict_result:", result)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alternatives.finder import find_fresh_alternatives


class FreshAlternativeFinderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now_ts = 1768771200  # 2026-01-18 UTC
        self.redis_v6_chunk = {
            "id": "redis_6_cluster_001",
            "product": "redis",
            "topic": "cluster_configuration",
            "version": "6.2",
            "date_ts": 1640995200,
            "metadata": {},
        }
        self.redis_v8_chunk = {
            "id": "redis_8_cluster_001",
            "product": "redis",
            "topic": "cluster_configuration",
            "version": "8.0",
            "date_ts": 1735689600,
            "metadata": {"status": "current"},
        }
        self.corpus = [self.redis_v6_chunk, self.redis_v8_chunk]

    def test_returns_fresher_replacement_for_stale_chunk(self) -> None:
        result = find_fresh_alternatives(
            retrieved_chunks=[self.redis_v6_chunk],
            corpus=self.corpus,
            query="How do I configure Redis Cluster?",
            now_ts=self.now_ts,
        )

        self.assertEqual(
            result,
            [
                {
                    "instead_of": "redis_6_cluster_001",
                    "use": "redis_8_cluster_001",
                    "reason": "Same product/topic, newer version available: 6.2 -> 8.0",
                }
            ],
        )

    def test_returns_no_alternative_for_fresh_chunk(self) -> None:
        result = find_fresh_alternatives(
            retrieved_chunks=[self.redis_v8_chunk],
            corpus=self.corpus,
            query="How do I configure Redis Cluster?",
            now_ts=self.now_ts,
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

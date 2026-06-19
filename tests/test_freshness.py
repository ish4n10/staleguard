import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scorers.freshness import score_chunk_freshness


class FreshnessScoringTests(unittest.TestCase):
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

    def test_marks_chunk_stale_when_newer_same_topic_exists(self) -> None:
        result = score_chunk_freshness(
            chunk=self.redis_v6_chunk,
            query="How do I configure Redis Cluster?",
            corpus=[self.redis_v6_chunk, self.redis_v8_chunk],
            now_ts=self.now_ts,
        )

        print("\nstale_result:", result)
        self.assertEqual(result["verdict"], "STALE")
        self.assertEqual(result["newer_available"], "redis_8_cluster_001")

    def test_keeps_explicitly_requested_old_version_fresh(self) -> None:
        result = score_chunk_freshness(
            chunk=self.redis_v6_chunk,
            query="How do I configure Redis Cluster in Redis 6.2?",
            corpus=[self.redis_v6_chunk, self.redis_v8_chunk],
            now_ts=self.now_ts,
        )

        print("\nrequested_version_result:", result)
        self.assertEqual(result["verdict"], "FRESH")
        self.assertIsNone(result["newer_available"])


if __name__ == "__main__":
    unittest.main()

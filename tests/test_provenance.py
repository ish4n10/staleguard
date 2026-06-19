import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auditor import build_provenance_card


class ProvenanceCardTests(unittest.TestCase):
    def test_builds_card_for_conflicted_case(self) -> None:
        retrieved_chunks = [
            {"id": "redis_6_cluster_001"},
            {"id": "redis_8_cluster_001"},
        ]
        stale_chunks = [
            {
                "id": "redis_6_cluster_001",
                "verdict": "STALE",
                "newer_available": "redis_8_cluster_001",
            }
        ]
        fresh_alternatives = [
            {
                "instead_of": "redis_6_cluster_001",
                "use": "redis_8_cluster_001",
                "reason": "Same product/topic, newer version available: 6.2 -> 8.0",
            }
        ]
        conflicts = [
            {
                "chunk_a": "redis_6_cluster_001",
                "chunk_b": "redis_8_cluster_001",
                "type": "VERSION_CONFLICT",
                "confidence": 0.75,
            }
        ]

        result = build_provenance_card(
            verdict="MIXED",
            confidence=0.71,
            retrieved_chunks=retrieved_chunks,
            stale_chunks=stale_chunks,
            fresh_alternatives=fresh_alternatives,
            conflicts=conflicts,
        )

        print("\nprovenance_card_result:", result)
        self.assertEqual(result["verdict"], "MIXED")
        self.assertEqual(result["confidence"], 0.71)
        self.assertEqual(result["total_chunks_checked"], 2)
        self.assertEqual(result["stale_count"], 1)
        self.assertEqual(result["conflict_count"], 1)
        self.assertEqual(result["used_chunks"], ["redis_6_cluster_001", "redis_8_cluster_001"])
        self.assertEqual(result["stale_chunks"], stale_chunks)
        self.assertEqual(result["fresh_alternatives"], fresh_alternatives)
        self.assertEqual(result["conflicts"], conflicts)
        self.assertEqual(
            result["recommendation"],
            "Block generation or ask user to resolve conflicting chunks.",
        )


if __name__ == "__main__":
    unittest.main()

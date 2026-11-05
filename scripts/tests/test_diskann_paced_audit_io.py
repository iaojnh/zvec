"""Reject synthetic throughput and preserve chronological artifact selection."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_diskann_paced_results import ordered_artifacts, verify_clock


class NativePacingAudit(unittest.TestCase):
    def test_throughput_must_use_complete_elapsed_time(self):
        query = {"queries": 1000, "repetitions": 3, "timed_block_wall_ms": 3000,
                 "qps": 1000, "search_only_qps": 1001}
        verify_clock(query)
        changed = copy.deepcopy(query)
        changed["qps"] = 5000
        with self.assertRaisesRegex(ValueError, "actual timed-block"):
            verify_clock(changed)
        query["native_pacing"] = {"reads": 10, "bytes": 40960, "actual_wait_ns": 4000000000}
        with self.assertRaisesRegex(ValueError, "actual wait"):
            verify_clock(query)

    def test_artifacts_are_ordered_by_creation_not_performance(self):
        artifacts = [{"id": 3, "created_at": "2026-11-05T08:00:00Z", "qps": 9000},
                     {"id": 1, "created_at": "2026-11-05T07:00:00Z", "qps": 100},
                     {"id": 2, "created_at": "2026-11-05T07:00:00Z", "qps": 5000}]
        self.assertEqual([1, 2, 3], [a["id"] for a in ordered_artifacts(artifacts)])


if __name__ == "__main__":
    unittest.main()

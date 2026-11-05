"""Guard total budgets, real concurrent wall time and live-thread RSS sampling."""
import copy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_diskann_concurrent_io import validate_concurrent_query
from summarize_diskann_concurrent_io import baseline_file_hashes


class ConcurrentNativeGuards(unittest.TestCase):
    def test_windows_checkout_line_endings_preserve_baseline_identity(self):
        original = b'[{"recall":0.9867}]\n'
        windows = original.replace(b'\n', b'\r\n')
        self.assertEqual(baseline_file_hashes(original), baseline_file_hashes(windows))
        self.assertIn(hashlib.sha256(windows).hexdigest(), baseline_file_hashes(original))
        self.assertNotIn(hashlib.sha256(original.replace(b'0.9867', b'0.9999')).hexdigest(),
                         baseline_file_hashes(original))

    def setUp(self):
        self.fixture = {"budgets": {"diskann": 32}, "recall_at_10": {"diskann": 0.9867}}
        self.query = {"fixed_budget": True, "budget": 32, "recall_at_10": 0.9867,
                      "queries": 1000, "repetitions": 3, "query_threads": 8,
                      "workers_alive_at_rss_sample": 8, "worker_queries": [375] * 8,
                      "max_active_queries": 8, "qps": 40, "timed_block_wall_ms": 75000,
                      "io": {"timed_read_bytes": 614400000},
                      "native_pacing": {"kind": "real-time-native-file-io", "scope": "process-wide",
                                        "resident_backing_bytes": 0, "iops_limit": 2000,
                                        "bytes_per_second_limit": 16777216, "max_in_flight": 8,
                                        "max_reserved_reads": 8, "reads": 150000, "bytes": 614400000,
                                        "achieved_iops": 2000, "actual_wait_ns": 300000000000}}

    def validate(self, query):
        validate_concurrent_query(query, "diskann", self.fixture, 2000, "linux-x86_64", 8, 16777216, 8)

    def test_aggregate_wait_can_exceed_wall_time_for_parallel_workers(self):
        self.validate(self.query)

    def test_per_worker_rate_or_depth_multiplication_is_rejected(self):
        for key, value in (("iops_limit", 16000), ("max_reserved_reads", 64), ("scope", "per-thread")):
            query = copy.deepcopy(self.query)
            query["native_pacing"][key] = value
            with self.assertRaises(ValueError):
                self.validate(query)

    def test_per_worker_iops_average_cannot_hide_total_reads(self):
        query = copy.deepcopy(self.query)
        query["native_pacing"]["reads"] *= 8
        query["native_pacing"]["bytes"] *= 8
        query["io"]["timed_read_bytes"] *= 8
        with self.assertRaisesRegex(ValueError, "not shared"):
            self.validate(query)

    def test_increasing_total_work_or_sampling_after_workers_exit_is_rejected(self):
        for key, value in (("worker_queries", [3000] * 8), ("workers_alive_at_rss_sample", 0),
                           ("qps", 320)):
            query = copy.deepcopy(self.query)
            query[key] = value
            with self.assertRaises(ValueError):
                self.validate(query)

    def test_idle_limiter_requires_workload_below_the_cap(self):
        query = copy.deepcopy(self.query)
        query["native_pacing"]["actual_wait_ns"] = 0
        with self.assertRaises(ValueError):
            self.validate(query)
        query["native_pacing"]["achieved_iops"] = 1500
        query["timed_block_wall_ms"] = 100000
        query["qps"] = 30
        self.validate(query)


if __name__ == "__main__":
    unittest.main()

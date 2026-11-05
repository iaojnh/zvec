"""Require real waiting, native storage evidence, and honest capacity tiers."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_diskann_paced_io import evaluate_probe, validate_query


def valid_probe():
    cases = []
    for state, actual, limit, observed in [
        ("buffered-warm", 100000, 0, 0),
        ("native-unlimited", 10000, 0, 33554432),
        ("paced-target", 1995, 2000, 33554432),
        ("paced-half", 998, 1000, 33554432),
    ]:
        cases.append({"state": state, "operations": 8192, "requested_bytes": 33554432,
                      "warmup_operations": 8192,
                      "seconds": 8192 / actual, "achieved_iops": actual, "iops_limit": limit,
                      "actual_wait_ns": 500000000 if limit else 0, "checksum": "1234567",
                      "cache_policy": "Linux native reader", "read_bytes": observed})
    return {"cases": cases}


class NativePacingGuards(unittest.TestCase):
    def test_real_rate_and_half_rate_are_required(self):
        self.assertTrue(evaluate_probe(valid_probe(), "linux-x86_64", 2000)["eligible"])
        report = valid_probe()
        report["cases"][2]["achieved_iops"] = 7000
        self.assertFalse(evaluate_probe(report, "linux-x86_64", 2000)["ceiling_verified"])
        report = valid_probe()
        report["cases"][2]["actual_wait_ns"] = 0
        self.assertFalse(evaluate_probe(report, "linux-x86_64", 2000)["eligible"])

    def test_slow_host_is_not_relabelled_as_high_tier(self):
        report = valid_probe()
        report["cases"][2]["achieved_iops"] = 1400
        self.assertFalse(evaluate_probe(report, "linux-x86_64", 2000)["tier_attainable"])
        with self.assertRaisesRegex(ValueError, "requested IOPS"):
            evaluate_probe(valid_probe(), "linux-x86_64", 10000)
        report = valid_probe()
        del report["cases"][2]["warmup_operations"]
        with self.assertRaisesRegex(ValueError, "volume changed"):
            evaluate_probe(report, "linux-x86_64", 2000)

    def test_cached_native_reads_and_changed_payload_are_rejected(self):
        report = valid_probe()
        report["cases"][1]["read_bytes"] = 0
        self.assertFalse(evaluate_probe(report, "linux-x86_64", 2000)["cache_verified"])
        report = valid_probe()
        report["cases"][2]["checksum"] = "different"
        with self.assertRaisesRegex(ValueError, "payload"):
            evaluate_probe(report, "linux-x86_64", 2000)

    def test_ios_host_observations_override_unavailable_sdk_counter(self):
        report = valid_probe()
        report["host_probe_io"] = {c["state"]: {"read_bytes": c["read_bytes"]} for c in report["cases"]}
        for case in report["cases"]:
            case["read_bytes"] = None
        self.assertTrue(evaluate_probe(report, "ios-arm64-simulator", 2000)["eligible"])

    def test_ram_backing_and_changed_query_budget_are_rejected(self):
        fixture = {"budgets": {"diskann": 32}, "recall_at_10": {"diskann": 0.9867}}
        query = {"fixed_budget": True, "budget": 32, "recall_at_10": 0.9867,
                 "native_pacing": {"kind": "real-time-native-file-io", "resident_backing_bytes": 0,
                                   "iops_limit": 2000, "reads": 50, "bytes": 204800,
                                   "achieved_iops": 1980, "actual_wait_ns": 20000000},
                 "io": {"timed_read_bytes": 204800}}
        validate_query(query, "diskann", fixture, 2000, "linux-x86_64")
        changed = copy.deepcopy(query)
        changed["native_pacing"]["resident_backing_bytes"] = 100000000
        with self.assertRaisesRegex(ValueError, "preloaded"):
            validate_query(changed, "diskann", fixture, 2000, "linux-x86_64")
        query["budget"] = 64
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_query(query, "diskann", fixture, 2000, "linux-x86_64")


if __name__ == "__main__":
    unittest.main()

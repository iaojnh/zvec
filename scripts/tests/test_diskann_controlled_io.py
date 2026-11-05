"""Guard against silently comparing different fixtures or storage models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_diskann_controlled_io import hashes, verify_fixture
from summarize_diskann_controlled_io import EXPECTED, summarize


class ControlledBenchmarkGuards(unittest.TestCase):
    def test_modified_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hnsw.index").write_bytes(b"hnsw")
            (root / "diskann.index").write_bytes(b"diskann")
            (root / "controlled-fixture.json").write_text(json.dumps({"sha256": hashes(root)}))
            verify_fixture(root)
            (root / "diskann.index").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_fixture(root)

    def reports(self, root):
        template = {
            "kind": "controlled-storage-model", "fixture": {"sha256": {"index": "hash"}},
            "model": {"latency_ns": 200000},
            "measurements": {"diskann": {
                "queries": 1000, "repetitions": 3, "recall_at_10": 0.98,
                "p95_ms": 0.1, "qps": 10000,
                "controlled_io": {"reads": 150000, "bytes": 614400000,
                                  "io_only_p95_ms": 1, "io_only_qps": 1000},
            }},
        }
        for label in EXPECTED:
            report = copy.deepcopy(template)
            report["label"] = label
            folder = root / label
            folder.mkdir()
            (folder / "controlled-results.json").write_text(json.dumps(report))

    def test_summary_requires_all_eight_and_marks_simulators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.reports(root)
            markdown, missing = summarize(root)
            self.assertFalse(missing)
            self.assertEqual(markdown.count("**SIMULATOR**"), 2)
            (root / "linux-arm64" / "controlled-results.json").unlink()
            markdown, missing = summarize(root)
            self.assertEqual(missing, ["linux-arm64"])
            self.assertIn("Missing/failed", markdown)

    def test_summary_rejects_different_fixture_or_model(self):
        for key, value in [("fixture", {"sha256": {"index": "other"}}),
                           ("model", {"latency_ns": 100000})]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.reports(root)
                path = root / "linux-arm64" / "controlled-results.json"
                report = json.loads(path.read_text())
                report[key] = value
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "Different canonical"):
                    summarize(root)


if __name__ == "__main__":
    unittest.main()

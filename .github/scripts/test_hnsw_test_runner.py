"""Reject false coverage reports without needing a device or native build."""

# The device tooling and its tests deliberately use only the standard library.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import hnsw_test_runner as runner


class CoverageChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.report = self.directory / "results.xml"

    def xml(self):
        root = ET.Element("testsuites", tests=str(runner.MINIMUM_TESTS))
        suite = ET.SubElement(root, "testsuite")
        for index in range(runner.MINIMUM_TESTS):
            ET.SubElement(suite, "testcase", name=str(index), status="run")
        properties = ET.SubElement(suite[0], "properties")
        ET.SubElement(
            properties,
            "property",
            name="fp16_cpu_architectures",
            value="NEON=1;Scalar=1;",
        )
        return root

    def verify(self, root):
        ET.ElementTree(root).write(self.report)
        return runner.verify_xml(self.report, "NEON")

    def console(self):
        count = runner.MINIMUM_TESTS
        return (
            "FP16 architecture coverage: NEON=1;Scalar=1;\n"
            f"[==========] {count} tests from 2 test suites ran. (100 ms total)\n"
            f"[  PASSED  ] {count} tests.\n"
        )

    def test_complete_xml(self):
        self.assertEqual(runner.MINIMUM_TESTS, self.verify(self.xml()))

    def test_xml_missing_case(self):
        root = self.xml()
        root[0].remove(root[0][-1])
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_skipped_case(self):
        root = self.xml()
        ET.SubElement(root[0][-1], "skipped")
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_legacy_skip_attribute(self):
        # GTest 1.10 uses result="skipped" without a <skipped> child.
        root = self.xml()
        root[0][-1].set("result", "skipped")
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_failed_case(self):
        root = self.xml()
        ET.SubElement(root[0][-1], "failure")
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_disabled_case(self):
        root = self.xml()
        root[0][-1].set("status", "notrun")
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_missing_architecture_record(self):
        root = self.xml()
        root[0][0].remove(root[0][0][0])
        with self.assertRaises(ValueError):
            self.verify(root)

    def test_xml_wrong_architecture(self):
        self.verify(self.xml())
        with self.assertRaises(ValueError):
            runner.verify_xml(self.report, "SSE2")

    def test_complete_console(self):
        self.assertEqual(runner.MINIMUM_TESTS, runner.verify_console(self.console()))

    def test_empty_console_suite(self):
        output = self.console().replace(str(runner.MINIMUM_TESTS), "0")
        with self.assertRaises(ValueError):
            runner.verify_console(output)

    def test_console_skipped_case(self):
        with self.assertRaises(ValueError):
            runner.verify_console(self.console() + "[  SKIPPED ] missing ISA\n")

    def test_console_failed_case(self):
        with self.assertRaises(ValueError):
            runner.verify_console(self.console() + "[  FAILED  ] broken graph\n")

    def test_console_incomplete_run(self):
        with self.assertRaises(ValueError):
            runner.verify_console("[==========] Running 73 tests from 2 test suites.\n")

    def test_console_missing_architecture(self):
        with self.assertRaises(ValueError):
            runner.verify_console("\n".join(self.console().splitlines()[1:]))

    def test_failed_process_overrides_passing_output(self):
        result = subprocess.CompletedProcess(["device"], 9, stdout=self.console())
        with (
            mock.patch.object(runner.subprocess, "run", return_value=result),
            mock.patch.object(runner.sys, "stdout", io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            runner.run(["device"], self.directory / "run.log")

    def test_timeout_preserves_partial_log(self):
        error = subprocess.TimeoutExpired(["device"], 1, output=b"partial output\n")
        with (
            mock.patch.object(runner.subprocess, "run", side_effect=error),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            runner.run(["device"], self.directory / "timeout.log")
        self.assertIn("partial output", (self.directory / "timeout.log").read_text())

    def test_android_rejects_wrong_abi(self):
        binary = self.directory / "hnsw_platform_test"
        binary.touch()
        with (
            mock.patch.object(runner, "run", return_value="x86_64\n") as command,
            self.assertRaises(ValueError),
        ):
            runner.android("device", binary, self.directory)
        self.assertEqual(1, command.call_count)

    def test_android_rejects_emulator(self):
        binary = self.directory / "hnsw_platform_test"
        binary.touch()
        with (
            mock.patch.object(runner, "run", side_effect=["arm64-v8a\n", "1\n"]),
            self.assertRaises(ValueError),
        ):
            runner.android("device", binary, self.directory)

    def test_android_cleanup_timeout_preserves_original_failure(self):
        binary = self.directory / "hnsw_platform_test"
        binary.touch()
        failure = subprocess.CalledProcessError(9, ["device"])
        with (
            mock.patch.object(
                runner, "run", side_effect=["arm64-v8a\n", "0\n", "", "", failure]
            ),
            mock.patch.object(
                runner.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["cleanup"], 30),
            ),
            self.assertLogs(level="WARNING"),
            self.assertRaises(subprocess.CalledProcessError) as error,
        ):
            runner.android("device", binary, self.directory)
        self.assertIs(error.exception, failure)

    def test_ios_requires_macos(self):
        with (
            mock.patch.object(sys, "platform", "linux"),
            self.assertRaises(ValueError),
        ):
            runner.ios("device", self.directory, self.directory)


if __name__ == "__main__":
    unittest.main()

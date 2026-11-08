# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Run the same budget, storage and public API contracts on each Linux backend."""

# ruff: noqa: T201, PT018, PLC0415
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

NATIVE_TARGETS = (
    "process_rss_budget_test",
    "pool_admission_test",
    "result_memory_allowance_test",
    "write_buffer_reservation_test",
    "retained_block_allocator_test",
    "budgeted_mapped_buffer_test",
    "eviction_queue_budget_test",
    "slab_backing_budget_test",
    "budgeted_aligned_buffer_test",
    "vector_page_table_test",
    "vector_page_table_bypass_aio_test",
    "buffer_read_storage_test",
    "buffer_storage_write_test",
    "ivf_workspace_budget_test",
    "diskann_file_reader_aio_test",
    "diskann_filtered_beam_test",
    "index_build_admission_test",
    "c_api_memory_budget_test",
    "internal_ids_memory_test",
    "flat_flow_workspace_test",
    "flat_streamer_buffer_test",
    "turbo_pq_int8_training_memory_test",
    "pq_buffered_training_test",
    "ivf_buffered_build_test",
    "opt_kmeans_cluster_test",
    "stratified_cluster_storage_test",
    "search_heap_test",
    "fts_column_indexer_test",
    "bitpacked_posting_list_test",
    "logger_test",
)
PYTHON_TESTS = (
    "test_arrow_executor_budget.py",
    "test_ddl_input_budget.py",
    "test_query_input_budget.py",
    "test_write_input_budget.py",
    "test_scan_input_budget.py",
    "test_sequence_input_budget.py",
    "test_fetch_result_budget.py",
)
BACKENDS = {"pread": 0, "libaio": 1, "io_uring": 2}
PAGE_CYCLE_TESTS = (
    "EvictedReadOnlyPageReloadsFromDisk",
    "EvictedDirtyPageReloadsPersistedBytes",
)


def verify_page_cycles(tests):
    for name in PAGE_CYCLE_TESTS:
        matching = [
            test
            for test in tests
            if test.get("classname") == "BufferPoolTest" and test.get("name") == name
        ]
        assert len(matching) == 1 and all(
            matching[0].find(tag) is None for tag in ("skipped", "failure", "error")
        ), f"missing successful page eviction/reload: {name}"


def verify_backend(platform_name, backend, actual, system, machine):
    expected_machine = {"linux-x64": "x86_64", "linux-arm64": "aarch64"}
    assert platform_name in expected_machine
    assert system == "linux" and machine == expected_machine[platform_name], (
        "wrong native runner architecture"
    )
    assert actual == BACKENDS[backend], f"expected {backend}, got backend {actual}"


def verify_native_result(target, output, xml):
    if target == "c_api_memory_budget_test":
        assert "C API memory admission and recovery passed" in output, (
            "missing C API completion"
        )
        suite = ET.Element("testsuite", name=target, tests="1", failures="0")
        ET.SubElement(suite, "testcase", name="memory_admission_and_recovery")
        ET.ElementTree(suite).write(xml, encoding="unicode")
    root = ET.parse(xml).getroot()
    tests = list(root.iter("testcase"))
    assert tests and any(t.find("skipped") is None for t in tests), (
        "no native tests executed"
    )
    assert not any(
        t.find("failure") is not None or t.find("error") is not None for t in tests
    )
    if target == "vector_page_table_test":
        verify_page_cycles(tests)
    return root


def run_command(command, cwd, log, timeout=900):
    with log.open("w") as output:
        try:
            result = subprocess.run(
                command,
                check=False,
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            print(f"\nTimed out after {timeout} seconds", file=output)
            return 124


def run(args):
    import zvec

    root = Path(__file__).resolve().parents[2]
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    actual = int(zvec.io_backend_type())
    record = {
        "platform": args.platform,
        "expected_backend": args.backend,
        "actual_backend": actual,
        "native_targets": list(NATIVE_TARGETS),
        "python_tests": list(PYTHON_TESTS),
        "passed": False,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    record_path = evidence / "backend.json"
    record_path.write_text(json.dumps(record, indent=2))
    verify_backend(
        args.platform, args.backend, actual, sys.platform, platform.machine()
    )
    failures = []
    combined = ET.Element("testsuites")
    with tempfile.TemporaryDirectory(prefix="zvec-backend-") as temporary:
        for target in NATIVE_TARGETS:
            binary = (args.native_bin / target).resolve()
            work = Path(temporary) / target
            work.mkdir()
            xml = evidence / (target + ".xml")
            command = [
                str(binary),
                "--gtest_color=no",
                "--gtest_output=xml:" + str(xml),
            ]
            log = evidence / (target + ".log")
            try:
                assert binary.is_file(), f"missing native target {binary}"
                status = run_command(command, work, log)
                assert status == 0, f"{target} exited {status}"
                combined.append(verify_native_result(target, log.read_text(), xml))
            except (AssertionError, OSError, ET.ParseError) as error:
                failures.append(str(error))
                suite = ET.SubElement(combined, "testsuite", name=target)
                case = ET.SubElement(suite, "testcase", name="execution")
                ET.SubElement(case, "failure").text = str(error)
                print(target, "FAILED:", str(error), flush=True)
            else:
                print(target, "passed", flush=True)
    ET.ElementTree(combined).write(evidence / "native.xml", encoding="unicode")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-o",
        "addopts=",
        "--junitxml=" + str(evidence / "python.xml"),
    ]
    command += [str(root / "python/tests" / name) for name in PYTHON_TESTS]
    if run_command(command, root, evidence / "python.log", timeout=1800) != 0:
        failures.append("public Python budget contracts failed")
    record.update(passed=not failures, failures=failures)
    record_path.write_text(json.dumps(record, indent=2))
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--platform")
    parser.add_argument("--native-bin", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.print_targets:
        print(" ".join(NATIVE_TARGETS))
    else:
        if not all((args.backend, args.platform, args.native_bin, args.evidence)):
            parser.error("backend, platform, native-bin and evidence are required")
        run(args)

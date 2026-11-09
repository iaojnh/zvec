# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Keep the failing WAL recovery collection and compare read paths after failure.

The pytest plugin observes the original test; it does not retry it or change its
assertions. The standalone probe repeats that same complete test, stops at the
first failure, and records native INFO logs in a separate process from the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import traceback
import zipfile
from pathlib import Path

import pytest

CASE = "test_pending_wal_requires_writable_recovery"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


def _read_paths(collection):
    result = {}
    # An iterator seals a writable segment. Only inspect additional paths on
    # the read-only handle, after the original assertion has already failed.
    try:
        result["read_only"] = collection.option.read_only
    except Exception as error:
        return {"options_error": repr(error)}
    if not result["read_only"]:
        return result
    probes = {
        "stats": lambda: repr(collection.stats),
        "legacy_fetch": lambda: repr(collection._obj.Fetch(["a", "b", "c"])),
        "single_fetch": lambda: {
            key: repr(collection.fetch(key)) for key in ("a", "b", "c")
        },
        "iterator": lambda: [repr(doc) for doc in collection.iter_docs()],
    }
    for name, probe in probes.items():
        try:
            result[name] = probe()
        except Exception as error:
            result[name] = {"error": repr(error)}
    return result


def capture_failure(output, nodeid, temporary, error):
    """Best effort diagnostics; callers must preserve the original failure."""
    output.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(nodeid.encode()).hexdigest()[:12]
    stem = f"failure-{os.getpid()}-{token}"
    result = {
        "nodeid": nodeid,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "traceback": "".join(traceback.format_exception(error)),
        "files": {},
    }
    # Copy before any additional reads and before pytest releases the traceback
    # and its live collection. Only this tiny synthetic collection is retained.
    source = temporary / "collection"
    snapshots = {}
    collection = None
    frame = error.__traceback__
    while frame:
        local = frame.tb_frame.f_locals
        if frame.tb_frame.f_code.co_name == CASE:
            for name in ("before", "recovered"):
                if isinstance(local.get(name), dict):
                    snapshots[name] = local[name]
        if frame.tb_frame.f_code.co_name == "_assert_docs":
            collection = local.get("collection")
        frame = frame.tb_next
    total = 0
    with zipfile.ZipFile(output / f"{stem}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for stage, snapshot in snapshots.items():
            for name, (data, inode, mtime_ns) in snapshot.items():
                if len(data) > MAX_ARCHIVE_BYTES - total:
                    result["files"][f"{stage}/{name}"] = {"error": "archive size limit"}
                    continue
                archive.writestr(f"{stage}/{name}", data)
                total += len(data)
                result["files"][f"{stage}/{name}"] = {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "inode": inode,
                    "mtime_ns": mtime_ns,
                }
        for path in sorted(source.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            name = "after-failure/" + path.relative_to(source).as_posix()
            try:
                size = path.stat().st_size
                if size > MAX_ARCHIVE_BYTES - total:
                    result["files"][name] = {
                        "error": "archive size limit",
                        "size": size,
                    }
                    continue
                data = path.read_bytes()
                if len(data) > MAX_ARCHIVE_BYTES - total:
                    raise ValueError("file grew beyond archive size limit")
                archive.writestr(name, data)
                total += len(data)
                result["files"][name] = {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            except Exception as copy_error:
                result["files"][name] = {"error": repr(copy_error)}
    if collection is not None:
        result["read_paths_after_failure"] = _read_paths(collection)
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def pytest_addoption(parser):
    parser.addoption("--wal-diagnostics-dir", type=Path, default=None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    output = item.config.getoption("--wal-diagnostics-dir")
    if (
        output is None
        or call.when != "call"
        or not report.failed
        or item.originalname != CASE
        or Path(str(item.path)).name != "test_readonly_storage.py"
    ):
        return
    try:
        capture_failure(
            output, item.nodeid, item.funcargs["tmp_path"], call.excinfo.value
        )
        token = hashlib.sha256(item.nodeid.encode()).hexdigest()[:12]
        (output / f"capture-{os.getpid()}-{token}.log").write_text(
            report.capstdout + report.capstderr, encoding="utf-8"
        )
    except Exception as error:
        report.sections.append(("WAL diagnostics error", repr(error)))


def run_probe(output, attempts, test_path):
    zvec = importlib.import_module("zvec")

    output.mkdir(parents=True, exist_ok=True)
    native_logs = output / "native"
    native_logs.mkdir(exist_ok=True)
    # Match the original module fixture's defaults; only the logger differs.
    zvec.init(
        log_type=zvec.LogType.FILE,
        log_level=zvec.LogLevel.INFO,
        log_dir=str(native_logs.resolve()),
        log_file_size=256,
    )
    spec = importlib.util.spec_from_file_location("wal_recovery_test", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = {
        "attempts": attempts,
        "passed": 0,
        "test": str(test_path),
        "case": f"{CASE}[insert-scalar-True]",
    }
    try:
        for attempt in range(attempts):
            temporary = output / f"attempt-{attempt:03}"
            temporary.mkdir()
            try:
                getattr(module, CASE)(temporary, True, "scalar", "insert")
            except Exception as error:
                summary["failed_attempt"] = attempt
                try:
                    capture_failure(output, summary["case"], temporary, error)
                except Exception as diagnostic_error:
                    sys.stderr.write(
                        f"Diagnostic capture failed: {diagnostic_error!r}\n"
                    )
                    sys.stderr.flush()
                raise
            summary["passed"] += 1
            shutil.rmtree(temporary)
            sys.stdout.write(f"WAL recovery probe {attempt + 1}/{attempts} passed\n")
            sys.stdout.flush()
    finally:
        (output / "probe-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument(
        "--test", type=Path, default=Path("python/tests/test_readonly_storage.py")
    )
    args = parser.parse_args()
    if not 1 <= args.attempts <= 100:
        parser.error("attempts must be between 1 and 100")
    run_probe(args.output.resolve(), args.attempts, args.test.resolve())

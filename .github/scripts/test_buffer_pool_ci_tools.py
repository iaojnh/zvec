# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Exercise backend, device and diagnostic failure paths without CI hardware."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET

import pytest
from buffer_pool_backend_tests import run_command, verify_backend, verify_native_result
from collect_ci_evidence import collect
from run_mobile_pressure import validate_output


@pytest.mark.parametrize(
    ("name", "machine"), [("linux-x64", "x86_64"), ("linux-arm64", "aarch64")]
)
@pytest.mark.parametrize(
    ("backend", "actual"), [("io_uring", 2), ("libaio", 1), ("pread", 0)]
)
def test_real_backend_and_architecture_are_required(name, machine, backend, actual):
    verify_backend(name, backend, actual, "linux", machine)
    with pytest.raises(AssertionError):
        verify_backend(name, backend, (actual + 1) % 3, "linux", machine)
    with pytest.raises(AssertionError):
        verify_backend(name, backend, actual, "darwin", machine)
    with pytest.raises(AssertionError):
        verify_backend(name, backend, actual, "linux", "wrong")


def test_c_api_requires_its_own_completion_marker(tmp_path):
    xml = tmp_path / "c.xml"
    with pytest.raises(AssertionError):
        verify_native_result("c_api_memory_budget_test", "", xml)
    verify_native_result(
        "c_api_memory_budget_test", "C API memory admission and recovery passed", xml
    )
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 1


@pytest.mark.parametrize(
    "body",
    [
        "",
        '<testcase name="skipped"><skipped/></testcase>',
        '<testcase name="failed"><failure/></testcase>',
    ],
)
def test_native_result_cannot_hide_missing_or_skipped_execution(tmp_path, body):
    xml = tmp_path / "native.xml"
    xml.write_text("<testsuite>" + body + "</testsuite>")
    with pytest.raises(AssertionError):
        verify_native_result("pool_admission_test", "", xml)


def test_timeout_is_a_failure_and_preserves_output(tmp_path):
    log = tmp_path / "timeout.log"
    assert (
        run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, log, 0.01
        )
        == 124
    )
    assert "Timed out" in log.read_text()


def device_output(platform="android"):
    row = {
        "rss_bytes": 32 * 1024**2,
        "target_bytes": 128 * 1024**2,
        "footprint_bytes": 40 * 1024**2 if platform == "ios" else 0,
    }
    phases = ["background_reclaimed", "critical_denied", "normal_retry"] + [
        "closed"
    ] * 36
    evidence = ["MOBILE_RSS " + json.dumps({**row, "phase": phase}) for phase in phases]
    return "\n".join(
        [
            *evidence,
            "[==========] 3 tests from 1 test suite ran. (100 ms total)",
            "[  PASSED  ] 3 tests.",
        ]
    )


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_required_device_pressure_checkpoints(platform):
    assert len(validate_output(device_output(platform), platform)) == 39


@pytest.mark.parametrize(
    "damage", ["empty", "failure", "incomplete", "pressure", "rss", "simulator"]
)
def test_mobile_evidence_fails_closed(damage):
    output = device_output("ios")
    if damage == "empty":
        output = ""
    elif damage == "failure":
        output += "\n[  FAILED  ] late crash"
    elif damage == "incomplete":
        output = output.replace('"phase": "closed"', '"phase": "missing"', 1)
    elif damage == "pressure":
        output = output.replace("critical_denied", "missing")
    elif damage == "rss":
        output = output.replace('"rss_bytes": 33554432', '"rss_bytes": 536870912')
    else:
        output = output.replace('"footprint_bytes": 41943040', '"footprint_bytes": 0')
    with pytest.raises(AssertionError):
        validate_output(output, "ios")


def test_diagnostics_keep_reports_without_runner_secrets(tmp_path):
    build = tmp_path / "build"
    logs = build / "Testing/Temporary"
    logs.mkdir(parents=True)
    (logs / "LastTest.log").write_text("failed native test")
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nPRIVATE_TOKEN:STRING=secret\n"
    )
    output = tmp_path / "evidence"
    collect(build, output)
    assert (output / "LastTest.log").read_text() == "failed native test"
    assert "PRIVATE_TOKEN" not in (output / "build-config.txt").read_text()
    assert json.loads((output / "environment.json").read_text())["commit"]


def test_diagnostics_find_checkout_without_changing_global_git_config(
    tmp_path, monkeypatch
):
    global_config = tmp_path / "global-git-config"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "evidence"
    collect(tmp_path / "missing-build", output)
    assert json.loads((output / "environment.json").read_text())["commit"]
    assert not global_config.exists()

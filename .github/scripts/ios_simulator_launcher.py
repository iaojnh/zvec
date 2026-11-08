# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Run an iOS test, recovering only a stale simctl console FIFO collision."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from time import monotonic


def launch(command, env, timeout):
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.pid, process.returncode, output or ""
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            return process.pid, 124, (output or "") + "\nTest timed out.\n"


def remove_stale_console_fifo(device_id, pid, returncode, output):
    if (
        returncode != 17
        or "domain=NSPOSIXErrorDomain, code=17" not in output
        or pid <= 0
        or not re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device_id
        )
    ):
        return False
    # A test that started, failed, or crashed must retain its original result.
    if re.search(
        r"(?m)^(?:\[iOS\]|\[[ =-]*(?:RUN|OK|PASSED|FAILED|==========)|"
        r"Running main\(|(?:Passed|Failed):|C API memory admission)",
        output,
    ):
        return False
    match = re.search(r"(?m)^Unable to establish FIFO at (.+): Error 17\r?$", output)
    if not match:
        return False
    path = Path(match[1])
    expected = (
        Path.home() / "Library/Developer/CoreSimulator/Devices" / device_id / "data/tmp"
    )
    if path.parent != expected or not re.fullmatch(
        rf"launch_console-{pid}-[0-9]+", path.name
    ):
        return False
    try:
        info = path.lstat()
        if stat.S_ISFIFO(info.st_mode) and info.st_uid == os.getuid():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def run_test(device_id, bundle_id, log_file, *, timeout=600, runner=launch):
    env = os.environ.copy()
    env["SIMCTL_CHILD_GTEST_COLOR"] = "no"
    command = ["xcrun", "simctl", "launch", "--console", device_id, bundle_id]
    deadline = monotonic() + timeout
    outputs = []
    for attempt in range(2):
        remaining = deadline - monotonic()
        if remaining <= 0:
            outputs.append("\nTest timed out.\n")
            returncode = 124
            break
        pid, returncode, output = runner(command, env, remaining)
        outputs.append(output)
        if attempt or not remove_stale_console_fifo(device_id, pid, returncode, output):
            break
        outputs.append(
            "\n::warning::Removed the stale FIFO for this simctl launch; "
            "retrying before the test app started.\n"
        )
    output = "".join(outputs)
    Path(log_file).write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return returncode


if __name__ == "__main__":
    sys.exit(run_test(*sys.argv[1:]))

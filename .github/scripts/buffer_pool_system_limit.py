# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Run inside a cgroup v2 service; fail closed if its hard limits are missing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def snapshot(group):
    result = {}
    for name in (
        "memory.max",
        "memory.peak",
        "memory.current",
        "memory.swap.max",
        "memory.swap.current",
        "memory.events",
    ):
        value = (group / name).read_text().strip()
        result[name] = (
            dict(line.split() for line in value.splitlines())
            if name.endswith("events")
            else int(value)
        )
    return result


def current_cgroup(membership=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    relative = next(
        line.split(":", 2)[2]
        for line in membership.read_text().splitlines()
        if line.startswith("0::")
    )
    return mount / relative.lstrip("/")


def run(limit, evidence, command):
    group = current_cgroup()
    before = snapshot(group)
    assert before["memory.max"] == limit, "external memory limit was not applied"
    assert before["memory.swap.max"] == before["memory.swap.current"] == 0, (
        "swap must be disabled"
    )
    with evidence.open("x") as output:
        output.write(json.dumps({"phase": "cgroup_start", **before}) + "\n")
        output.flush()
        result = subprocess.run(command, check=False)
        after = snapshot(group)
        output.write(
            json.dumps(
                {"phase": "cgroup_complete", "returncode": result.returncode, **after}
            )
            + "\n"
        )
    assert result.returncode == 0, f"workload exited {result.returncode}"
    assert after["memory.swap.current"] == 0
    for event in ("oom", "oom_kill", "oom_group_kill"):
        assert int(after["memory.events"].get(event, 0)) == int(
            before["memory.events"].get(event, 0)
        ), event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limit_bytes", type=int)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command
    run(args.limit_bytes, args.evidence, args.command)

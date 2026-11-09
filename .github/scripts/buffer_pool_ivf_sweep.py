# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Run/export a declared memory sweep without treating failed points as profiles."""

# JSON and Markdown are the CI command-line contract.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from buffer_pool_ivf_profile import STAGES, evaluate, qualified_profile, read_case


def selected_config(template, budget):
    config = json.loads(json.dumps(template))
    if budget not in config["buffer_memory_limits_mb"]:
        raise ValueError(f"undeclared Buffer Pool memory limit: {budget}")
    config["memory_limit_mb"] = budget
    config["system_limit_mib"] = budget + config["buffer_system_headroom_mib"]
    config["name"] = (
        f"ivf-fp32-{config['rows'] // 1000}k-{config['dimension']}d-{budget}m"
    )
    return config


def configure(args):
    config = selected_config(json.loads(args.profile.read_text()), args.memory_limit_mb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(config, output, indent=2)
        output.write("\n")
    return 0


def lifecycle(root, config, digest):
    values = []
    for stage in STAGES:
        try:
            value = read_case(root, config, digest, "buffer", stage, 1)
            values.append(
                {
                    "stage": stage,
                    "passed": True,
                    "peak_rss_mib": value["rss_observation"]["peak_rss_kib"] / 1024,
                }
            )
        except (AssertionError, KeyError, ValueError, OSError) as error:
            values.append({"stage": stage, "passed": False, "error": str(error)})
    return values


def read_point(template, root, budget):
    selected = root / "profile-result/ivf_memory_profile.json"
    report = root / "profile-result/ivf-profile-report.json"
    point = {"memory_limit_mb": budget, "passed": False, "complete": False}
    try:
        config = json.loads(selected.read_text())
        if config != selected_config(template, budget):
            raise ValueError("sweep point configuration differs from declared controls")
        # A report proves this point reached export. A workload failure is a
        # useful negative measurement; an absent artifact is an incomplete run.
        json.loads(report.read_text())
        point["complete"] = True
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        point["buffer_lifecycle"] = lifecycle(root / "evidence", config, digest)
        point.update(evaluate(selected, root / "evidence"))
    except (AssertionError, KeyError, ValueError, OSError) as error:
        point["error"] = f"{type(error).__name__}: {error}"
    return point


def markdown(result):
    lines = [
        "# IVF memory sweep",
        "",
        f"mmap uses {result['mmap_reference_memory_limit_mb']} MiB in every stage; swap is disabled.",
        "Each Buffer Pool point runs build, sustained writes with queries, reopen and three query trials.",
        "Ratios use QPS / time-weighted whole-process RSS against its same-runner mmap reference.",
        "QPS and p99 are reported in the point reports; no raw-QPS floor replaces the QPS/RSS goal.",
        "",
        "| Buffer limit MiB | External MiB | Lifecycle | Peak lifecycle RSS MiB | QPS/RSS ratios | Qualified |",
        "|---:|---:|---|---:|---|---|",
    ]
    for point in result["points"]:
        phases = point.get("buffer_lifecycle", [])
        peaks = [phase["peak_rss_mib"] for phase in phases if phase["passed"]]
        lifecycle_passed = len(phases) == len(STAGES) and all(
            phase["passed"] for phase in phases
        )
        ratios = ", ".join(
            f"{case['distribution']}: {case['qps_per_rss_ratio']:.3f}"
            for case in point.get("comparisons", [])
        )
        peak = f"{max(peaks):.1f}" if peaks else "—"
        external = point["memory_limit_mb"] + result["buffer_system_headroom_mib"]
        lines.append(
            f"| {point['memory_limit_mb']} | {external} | {'PASS' if lifecycle_passed else 'FAIL'} "
            f"| {peak} | {ratios or 'unavailable'} | {'PASS' if point['passed'] else 'FAIL'} |"
        )
    for point in result["points"]:
        if "error" in point:
            lines += ["", f"{point['memory_limit_mb']} MiB: {point['error']}"]
    if result["passed"]:
        lines += [
            "",
            f"Lowest qualified limit: **{result['selected_memory_limit_mb']} MiB**. "
            "`ivf-profile-qualified.json` is directly runnable with the public example.",
        ]
    else:
        lines += ["", "No qualified deployment profile exported."]
    return "\n".join([*lines, ""])


def collect(args):
    template = json.loads(args.profile.read_text())
    points = [
        read_point(template, args.root / f"buffer-pool-ivf-profile-{budget}m", budget)
        for budget in template["buffer_memory_limits_mb"]
    ]
    qualified = sorted(
        (point for point in points if point["passed"]),
        key=lambda point: point["memory_limit_mb"],
    )
    signatures = {
        (point["source_commit"], point["wheel_sha256"])
        for point in points
        if "source_commit" in point
    }
    complete = all(point["complete"] for point in points) and len(signatures) <= 1
    result = {
        "passed": complete and bool(qualified),
        "complete": complete,
        "mmap_reference_memory_limit_mb": template["mmap_reference_memory_limit_mb"],
        "buffer_system_headroom_mib": template["buffer_system_headroom_mib"],
        "points": points,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "ivf-profile-qualified.json"
    receipt.unlink(missing_ok=True)
    if result["passed"]:
        budget = qualified[0]["memory_limit_mb"]
        result["selected_memory_limit_mb"] = budget
        # Build the receipt from revalidated evidence, never from a possibly
        # stale qualification file in the downloaded artifact.
        receipt.write_text(json.dumps(qualified_profile(qualified[0]), indent=2) + "\n")
    (args.output / "ivf-sweep-report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output / "ivf-sweep-report.md").write_text(markdown(result))
    print(json.dumps({key: value for key, value in result.items() if key != "points"}))
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("configure")
    select.add_argument("--profile", type=Path, required=True)
    select.add_argument("--memory-limit-mb", type=int, required=True)
    select.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("collect")
    report.add_argument("--profile", type=Path, required=True)
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return configure(args) if args.command == "configure" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Sweep selection must preserve failures and never recommend incomplete evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import pytest
from buffer_pool_ivf_profile import export
from buffer_pool_ivf_sweep import collect, selected_config
from test_buffer_pool_ivf_profile import evidence  # noqa: F401


@pytest.fixture
def sweep(evidence, tmp_path, capsys):  # noqa: F811
    profile, original = evidence
    template = json.loads(profile.read_text())
    root = tmp_path / "points"
    for budget in template["buffer_memory_limits_mb"]:
        point = root / f"buffer-pool-ivf-profile-{budget}m"
        raw = point / "evidence"
        shutil.copytree(original, raw)
        config = selected_config(template, budget)
        selected = point / "profile-result/ivf_memory_profile.json"
        selected.parent.mkdir(parents=True)
        selected.write_text(json.dumps(config))
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        for path in raw.glob("*.json"):
            value = json.loads(path.read_text())
            value.update(profile=config, profile_sha256=digest)
            path.write_text(json.dumps(value))
        for path in raw.glob("buffer-*.monitor.jsonl"):
            value = json.loads(path.read_text())
            value["target_rss_kib"] = budget * 1024
            path.write_text(json.dumps(value))
        for path in raw.glob("buffer-*.cgroup.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["memory.max"] = config["system_limit_mib"] * 1024**2
            path.write_text("\n".join(json.dumps(row) for row in rows))
        assert (
            export(
                argparse.Namespace(
                    profile=selected, evidence=raw, output=selected.parent
                )
            )
            == 0
        )
        capsys.readouterr()
    return argparse.Namespace(profile=profile, root=root, output=tmp_path / "summary")


def point_root(args, budget):
    return args.root / f"buffer-pool-ivf-profile-{budget}m"


def remove_rss_advantage(args, budget):
    raw = point_root(args, budget) / "evidence"
    for path in raw.glob("buffer-query-*.monitor.samples.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["rss_kib"] = 400 * 1024
        path.write_text("\n".join(json.dumps(row) for row in rows))


def test_sweep_selects_lowest_qualified_budget_from_raw_evidence(sweep):
    # The aggregate recreates its receipt from validated raw evidence.
    (point_root(sweep, 256) / "profile-result/ivf-profile-qualified.json").unlink()
    assert collect(sweep) == 0
    report = json.loads((sweep.output / "ivf-sweep-report.json").read_text())
    assert len(report["points"]) == 5
    assert report["selected_memory_limit_mb"] == 256
    result = json.loads((sweep.output / "ivf-profile-qualified.json").read_text())
    assert result["memory_limit_mb"] == 256
    assert result["system_limit_mib"] == 512
    assert result["mmap_reference_memory_limit_mb"] == 8192


def test_nonqualifying_point_is_reported_but_not_selected(sweep):
    remove_rss_advantage(sweep, 256)
    assert collect(sweep) == 0
    report = json.loads((sweep.output / "ivf-sweep-report.json").read_text())
    assert report["complete"] is True
    assert report["points"][0]["passed"] is False
    assert report["selected_memory_limit_mb"] == 512


@pytest.mark.parametrize("damage", ["missing", "different_controls", "mixed_commit"])
def test_incomplete_or_incomparable_sweep_cannot_export_a_profile(sweep, damage):
    assert collect(sweep) == 0
    point = point_root(sweep, 1536)
    if damage == "missing":
        (point / "profile-result/ivf-profile-report.json").unlink()
    elif damage == "different_controls":
        path = point / "profile-result/ivf_memory_profile.json"
        value = json.loads(path.read_text())
        value["batch_size"] *= 2
        path.write_text(json.dumps(value))
    else:
        for path in (point / "evidence").glob("*.json"):
            value = json.loads(path.read_text())
            value["source_commit"] = "c" * 40
            path.write_text(json.dumps(value))
    assert collect(sweep) == 1
    assert not (sweep.output / "ivf-profile-qualified.json").exists()


def test_completed_sweep_with_no_advantage_exports_no_profile(sweep):
    for budget in (256, 512, 768, 1024, 1536):
        remove_rss_advantage(sweep, budget)
    assert collect(sweep) == 1
    report = json.loads((sweep.output / "ivf-sweep-report.json").read_text())
    assert report["complete"] is True
    assert not report["passed"]
    assert not (sweep.output / "ivf-profile-qualified.json").exists()


def test_configure_rejects_undeclared_budget(evidence):  # noqa: F811
    profile, _ = evidence
    with pytest.raises(ValueError, match="undeclared"):
        selected_config(json.loads(profile.read_text()), 999)


def test_failed_pressure_gate_prevents_recommending_a_performance_pass(sweep):
    pressure = sweep.root / "buffer-pool-ivf-pressure-512m"
    shutil.copytree(point_root(sweep, 512), pressure)
    sweep.pressure_profile = sweep.profile
    assert collect(sweep) == 0
    path = pressure / "evidence/buffer-verify-1.json"
    value = json.loads(path.read_text())
    value["passed"] = False
    value["error"] = "injected persistence failure"
    path.write_text(json.dumps(value))
    assert collect(sweep) == 1
    assert not (sweep.output / "ivf-profile-qualified.json").exists()

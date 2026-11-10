"""Validate same-version mmap/BP measurements and separate CPU-hint controls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
p.add_argument("--plan", default="formal-plan.json")
p.add_argument("--prefix", default="default")
a = p.parse_args()
root = a.root
manifest = json.loads((root / "manifest.json").read_text())
reference = json.loads((root / "inputs/reference-zvec.json").read_text())
truth = (root / "inputs/truth.ibin").read_bytes()
assert hashlib.sha256(truth).hexdigest() == manifest["input_sha256"]["/work/data/truth.ibin"]
rows, width = struct.unpack_from("<II", truth)
truth = struct.unpack_from("<" + "I" * (rows * width), truth, 8)
groups, runs = defaultdict(list), []
for c in json.loads((root / a.plan).read_text()):
    name, limit = c["name"], c["limit_mib"]
    if manifest.get("same_native_binary_both_engines") and c["native_binary"] != manifest.get("hint_binary"):
        assert c["native_binary"] == manifest["same_native_binary_both_engines"]
    v = json.loads((root / "raw" / (name + ".json")).read_text())
    m = json.loads((root / "meta" / (name + ".json")).read_text())
    assert v["complete"] and m["complete"] and m["validation_complete"] and m["returncode"] == 0
    assert v["engine"] == c["engine"] and v["nprobe"] == 12 and v["queries"] == 400
    assert v["query_ids"] == reference["query_ids"] and v["query_fingerprint"] == 14738181560880891675
    assert v["result_ids"] == reference["result_ids"]
    hits = sum(sum(x in truth[q * width:q * width + 10] for x in v["result_ids"][i * 10:i * 10 + 10])
               for i, q in enumerate(v["query_ids"]))
    assert v["recall_at_10"] == hits / 4000 == .9645
    for k in ["cgroup_before", "cgroup_after"]:
        assert m[k]["memory.max"] == limit << 20
        assert m[k]["memory.swap.max"] == m[k]["memory.swap.current"] == 0
        assert str(m[k]["cpuset.cpus.effective"]) == "1"
    assert m["cgroup_after"]["memory.peak"] <= limit << 20
    assert all(m["cgroup_after"]["memory.events"].get(k, 0) == 0 for k in ["oom", "oom_kill", "oom_group_kill"])
    assert all(x["resident_after"] == 0 for x in m["cold_inputs"])
    assert all(x["resident_after"] == 0 for x in json.loads((root / "meta" / (name + ".lance-cold.json")).read_text()))
    history = json.loads((root / "commands" / (name + ".isolation.json")).read_text())
    assert history and all(s["profiles"]["default"] == "Stopped" for s in history)
    mode = "bp" if c["engine"] == "buffer" else "mmap"
    if c["native_binary"] == manifest.get("hint_binary"):
        assert c["engine"] == "mmap"
        mode = "mmap_equal_hint"
    if mode == "bp":
        assert v["cache_backend"] == "ChunkReadStorage" and v["direct_io_descriptors"] > 0
        cache = v["cache_after_measurement"]
        assert cache["prefetch_failed"] == cache["prefetch_pending_requests"] == cache["prefetch_pending_bytes"] == 0
        assert cache["prefetch_peak_requests"] <= 12 and cache["prefetch_peak_bytes"] <= c["read_ahead_bytes"]
    else:
        assert v["cache_backend"] != "ChunkReadStorage" and v["direct_io_descriptors"] == 0
    row = dict(name=name, limit_mib=limit, mode=mode, repeat=c["repeat"], qps=v["qps"],
               p50_ms=v["p50_ms"], p95_ms=v["p95_ms"], p99_ms=v["p99_ms"],
               wall_ms=1000 / v["qps"], user_ms=v["user_cpu_s"] * 2.5,
               system_ms=v["system_cpu_s"] * 2.5, read_gib=v["os_disk_read_bytes"] / 2**30,
               major_faults=v["os_pageins"], minor_faults=v["os_minor_faults"],
               rss_mib=v["rss_mean_mib"], managed_peak_mib=v["managed_peak_mib"],
               cgroup_peak_mib=m["cgroup_after"]["memory.peak"] / 2**20,
               anon_thp_mib=m["cgroup_after"]["memory.stat"].get("anon_thp", 0) / 2**20,
               file_thp_mib=m["cgroup_after"]["memory.stat"].get("file_thp", 0) / 2**20)
    groups[(limit, mode)].append(row)
    runs.append(row)
summary = []
for (limit, mode), values in groups.items():
    assert len(values) == 3
    row = dict(limit_mib=limit, mode=mode, repeats=3, qps_values=[v["qps"] for v in values])
    for k in values[0]:
        if k not in ["name", "mode", "repeat"]:
            row[k] = statistics.median(v[k] for v in values)
    summary.append(row)
comparisons = []
for limit in sorted({v["limit_mib"] for v in summary}):
    by_mode = {v["mode"]: v for v in summary if v["limit_mib"] == limit}
    mm, bp = by_mode["mmap"], by_mode["bp"]
    row = dict(limit_mib=limit, bp_over_default_mmap=bp["qps"] / mm["qps"],
               bp_qps_gain_pct=(bp["qps"] / mm["qps"] - 1) * 100)
    if "mmap_equal_hint" in by_mode:
        hinted = by_mode["mmap_equal_hint"]
        row.update(mmap_hint_gain_pct=(hinted["qps"] / mm["qps"] - 1) * 100,
                   bp_over_equal_hint_mmap=bp["qps"] / hinted["qps"],
                   equal_hint_mmap_minus_bp_wall_ms=hinted["wall_ms"] - bp["wall_ms"])
    comparisons.append(row)
validation = dict(complete=True, valid_runs=len(runs), measured_operating_points=len(summary),
                  queries=400, warmup=200, nprobe=12, recall=.9645, ordered_top10_identical=True,
                  same_current_binary_default_modes=True, same_fixed_index=True,
                  inclusive_memory_limits_verified=True, swap_disabled=True, no_oom=True,
                  cold_inputs_verified=True, default_vm_stopped=True, host_isolation_guaranteed=False)
for name, value in [("summary", summary), ("per-run", runs), ("comparison", comparisons), ("validation", validation)]:
    (root / (a.prefix + "-" + name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(dict(summary=summary, comparisons=comparisons, validation=validation), indent=2))

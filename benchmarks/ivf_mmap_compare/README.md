# IVF mmap CPU-prefetch comparison

These scripts validate and summarize the 2026-11-10 Linux ARM64 experiment.
The product change is in `f105db309de690cd5efdba917cd85b5e70730cf6`.
It remains on an experimental branch because unconditional mmap CPU prefetch
regresses when the index does not fit in the memory limit.

## Workload and results

The workload contains 1 million 768-dimensional FP32 cosine vectors, 160 fixed
centroids, nprobe 12, the original 200 warmup queries and 400 measured queries.
Each fresh query container uses one guest CPU, a cgroup v2 total-memory limit
including file-cache and kernel charges, and no swap. The other Colima VM stays
stopped; the separately authorized host DiskANN job continues running.

Both zvec readers use the same native binary and index. Each operating point is
measured three times in an interleaved schedule. All 18 runs return identical
ordered Top10 IDs at Recall@10 96.45%, with no OOM.

| Memory limit | mmap with CPU prefetch, QPS | Buffer pool, QPS |
|---|---:|---:|
| 1 GiB | 7.09 | 74.08 |
| 2 GiB | 9.84 | 97.33 |
| 4 GiB | 152.64 | 141.84 |

Values are medians of three runs. At 4 GiB the readers perform similarly; the
ranges overlap. At 1/2 GiB, mmap user CPU time increases substantially while
read volume stays close to recent measurements without the hint. The precise
microarchitectural cause has not been established with hardware counters.
The earlier dedicated 4 GiB control cohort measured default mmap at 98.58,
mmap with the same CPU hint as BP at 157.29, and BP at 154.01 QPS. Those values
belong to a separate cohort and must not be mixed with this table to calculate
a same-cohort code improvement.

## Inputs and usage

The scripts use Python's standard library. They consume the frozen benchmark
artifacts from the existing workspace; they do not generate the dataset, truth,
indexes, containers, or build environment. Large data and result artifacts are
not included in this commit.

Place the original result directories under `benchmarks/results/`, preserving
their names and relative layout. The primary directory is
`ivf-mmap-prefetch-linux-cgroup-20261110`, containing `manifest.json`,
`formal-plan.json`, `raw/`, `meta/`, `commands/`, and `inputs/`. The JSON files
record input/executable hashes, original query order, per-query result IDs,
memory limits, cold-input checks and the VM isolation history.

```sh
python3 benchmarks/ivf_mmap_compare/analyze.py \
  benchmarks/results/ivf-mmap-prefetch-linux-cgroup-20261110 \
  --plan formal-plan.json --prefix current
python3 benchmarks/ivf_mmap_compare/report_prefetch.py \
  benchmarks/results/ivf-mmap-prefetch-linux-cgroup-20261110
python3 benchmarks/ivf_mmap_compare/consolidate_updated.py
```

- `analyze.py` verifies the 18 runs, recalculates recall, checks the common
  executable, ordered result IDs, cgroup limits, swap, OOM events, cold inputs,
  VM isolation and BP asynchronous IO bounds, then summarizes all three repeats.
- `report_prefetch.py` additionally requires the saved regression-validation
  and postflight records, plus the preceding
  `ivf-mmap-bp-current-linux-cgroup-20261110` audit. It retains the low-memory
  regressions and distinguishes recent background measurements from a strict
  same-cohort A/B.
- `consolidate_updated.py` requires the previous
  `ivf-five-readers-optimized-overview-20261110` overview and its referenced
  evidence directories. It checks their hashes and replaces both zvec columns
  with fresh measurements. SPANN/Faiss/LanceDB retain their historical cohorts.
  Faiss uses independently trained centroids and seconds/query interpolation
  to Recall@10 96.45%; its other metrics remain actual p11 measurements at
  96.575%. The script writes a new overview directory, preserving the old one.
- `build_hint_control.py` runs inside the prepared Linux VM. It requires the
  frozen `/work/build`, `/work/capacity-src-20261110` and
  `/work/capacity-build-v4-20261110` artifacts. It recompiles only the two-hint
  diagnostic change and relinks the original remaining objects. It is for
  reproducing the earlier 4 GiB control, not for building the full project.

The product validation comprised 97 passing regression tests and a macOS C++
syntax check. This native experiment does not substitute for full-project CI.

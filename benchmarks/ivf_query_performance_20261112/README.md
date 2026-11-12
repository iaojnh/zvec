# IVF query-path performance evidence

Linux aarch64, GCC 13.3, Release `-O3`, one query thread pinned to a virtual CPU,
1,000,000 FP32 vectors, 768 dimensions, cosine distance, top-k 10. The index is
fully resident before each process; there is no container CPU or memory quota.

The confirmation cohort has 10 alternating baseline/optimized pairs, 200 warmup
queries and 400 measured queries per run. At 160 centers and nprobe 12:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Median QPS | 84.78 | 138.19 |
| Pooled P99 | 20.046 ms | 9.653 ms |
| Queries above 20 ms | 41 / 4,000 | 2 / 4,000 |
| Recall@10 | 96.45% | 96.45% |

[Per-run statistics for all 120 formal runs](runs.csv) and
[cohort summaries](summary.json).

The statistics cover 48,000 measured queries across the center-count sweep,
fixed-160 experiments and confirmation cohort. This measures the full
query-path change on warm-cache unquantized FP32 data, not encoding-buffer
savings in isolation or performance under memory pressure.

Only aggregate metrics and per-run statistics are published here. Query IDs,
vectors, index contents, result IDs/scores, individual query timings and host
process information are not included in this branch.

The historical fixed cohort uses 10 alternating baseline/optimized pairs at
160 centers and nprobe 12. The center-count cohort uses 5 pairs at each of 8
endpoints. Each run opens the same immutable index for the two versions, warms
up with 200 queries and times the same next 400 queries in original order.
QPS is 400 divided by measured wall time including result validation and RSS
sampling. QPS is summarized by median across runs. Pooled latency percentiles
combine individual latencies from every run; the per-run percentile columns
in runs.csv should not be averaged to reconstruct a pooled percentile.

The index was fully resident before each process. Timed runs had zero OS disk
read bytes, zero major faults and no memory-quota events. The baseline and
optimized readers returned identical ordered top-10 IDs and scores at every
endpoint, as verified locally before exporting these aggregates.

Measured baseline: v0.8.1 (`30d662db4d2467937d843443074b34743bcc9baa`).
Measured optimized implementation: `4da94f5d4b69fc0a67ac3f74d5759d1f02a3fdc9`.
Upstream base: `d61530f2c9c601aee3490704717481ae6d176456`.
The baseline and upstream base have identical compiled core sources. The two
production files in the submission match the measured optimized implementation.

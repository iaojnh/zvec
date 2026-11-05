# Real-time native file I/O pacing

This is a separate experiment from the earlier discrete-event cost model.
It retains the production file reader, including its direct/unbuffered/cache
eviction policy. It never loads the graph/vector file into a RAM backing image.
The benchmark-only decorator admits native read batches through real-time IOPS
and byte-rate token buckets, then submits them to the original reader. Native
read completion, CPU work, pacing waits and timer overshoot all contribute to
the measured wall time. Windows uses a high-resolution waitable timer when
available; Apple uses a monotonic kqueue timer with NOTE_CRITICAL to minimize
timer coalescing, and Linux/Android use monotonic sleep_until. No virtual latency
is added to the measured QPS or P95.

Credits persist across queries and are bounded to one concurrency-sized burst.
Batches larger than the configured depth are split and native completion
indices are mapped back to the original requests. Native per-context resource
ownership remains with the original reader. The ordinary production build
contains no reader hook unless the benchmark option is enabled.

## Tiers and qualification

Defaults: 2,000 / 10,000 / 20,000 IOPS plus an unlimited native reference, with
eight in-flight requests. Each IOPS tier has a byte-rate ceiling of the larger
of 16 MiB/s or twice its 4 KiB request throughput. 2,000 IOPS is a conservative
common CI tier, not a claim about typical modern-PC SSD performance. The higher
tiers are measured only where the native path demonstrates sufficient capacity.
Limiting cannot make a slow disk faster.
The cap counts native 4 KiB file-read requests; device firmware or virtualization
may combine or amplify physical operations, so it is not a device-IOPS guarantee.

Each platform first runs 2,048 actual 4 KiB reads through a buffered warm
control, the unlimited production reader, the target rate, and half that rate.
Payload checksums must match. Timed wall throughput must be at most 105% and at
least 90% of the configured probe rate; target/half-rate scaling must be 1.7–2.3.
The target must record real waiting. Unsupported higher tiers stay explicitly
blank. Failure of the shared 2,000-IOPS tier withholds its publication row and
fails CI rather than silently lowering the requested tier.

Linux/Android and Apple use OS-accounted storage-read bytes compared with warm
buffered reads to validate page-cache avoidance. iOS uses its simulator host's
public libproc observation of the live app process. Windows retains
FILE_FLAG_NO_BUFFERING, but its process counters are logical rather than proof
of physical-device reads. Device/hypervisor caches are not fully excluded.
Android's positive control runs in an unencrypted directory of a verified
ephemeral GitHub-hosted emulator; it is not a physical-phone measurement.

## Query measurements

Reuse the byte-identical trained indexes, vector/query matrices and frozen
budgets in the canonical fixture from the earlier successful experiment. No
index rebuilding or tuning on held-out queries happens on replay hosts.

For every attainable tier, both HNSW and DiskANN run in fresh processes, with
one workload warmup, 1,000 held-out queries repeated three times, and three
trials. P95 measures actual C++ search-call elapsed time; QPS uses the full
timed-block wall time, including benchmark-loop bookkeeping. Report per-trial
metrics and their medians. Timed DiskANN reads must continue to show storage
I/O evidence, and measured IOPS cannot exceed the configured query ceiling.

RSS is total steady query-process RSS, or the Windows working set, in decimal
MB. It includes the small limiter instrumentation but no full graph backing
image. Unmapped kernel file cache is outside this metric. HNSW is warmed and
resident; DiskANN uses the native cache policy shown in each result. These are
serial C++ core-index results, not full Collection/Python API performance.
CPU/RAM/storage are still those of the specific runner; rate ceilings do not
make eight physically different hosts identical or establish an OS ranking.

```sh
python scripts/benchmark_diskann_paced_io.py \
  --binary build/bin/diskann_platform_benchmark_test \
  --data benchmark-data --output paced-results --label local \
  --profiles 2000,10000,20000,0
```

Build with `-DDISKANN_CONTROLLED_IO_BENCHMARK=ON` and run
`diskann_controlled_io_test` first. The new `DiskANN Paced Native I/O` workflow
runs all eight environments. The old cost-model workflow is now manual-only.
The summary emits `publication-comparison.md` with both algorithms' RSS,
Recall@10, P95 and QPS, simulator labels and cache conditions, plus combined
raw JSON. Raw probe and trial results remain available for audit.

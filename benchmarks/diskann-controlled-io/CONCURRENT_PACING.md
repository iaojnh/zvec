# Concurrent native I/O protocol

Run only 2, 4 and 8 closed-loop query workers. Reuse the complete one-worker
results already committed under `results/2026-11-05`; never run a new one-worker
query benchmark. Its runner instance differs from the current host and this
is explicitly marked in the comparison tables.

All workers in a process share one trained index, one synchronized IOPS bucket,
one byte-rate bucket and eight total read slots. Token reservations are made
under a short mutex; waiting and native I/O happen outside it. Timers are private
to each worker on Windows/Apple, so one worker cannot rearm another's timer.
The native reader retains separate pending batches and per-thread I/O contexts.
Global read slots are released as completions are reaped. FIFO depth admission
prevents a worker from monopolizing the shared queue.

Each trial processes the same 1,000 evaluation queries repeated three times
total. Work is partitioned across threads, rather than multiplied by thread
count. The same worker threads execute one workload warmup, the timed workload,
and remain alive while steady RSS is sampled. Thus thread contexts and scratch
buffers are included. Close/join workers before closing the shared index.

QPS is total completed queries divided by full process wall time. P95 pools the
actual search-call latencies from all workers, including admission/depth waits.
Summed wait time may exceed process wall time because workers wait concurrently.
Three fresh processes per algorithm/thread-count/tier supply median metrics.

The shared common tier remains 2,000 IOPS, with optional 10,000/20,000 tiers and
an unlimited native reference. All tiers retain eight total read slots. Optional
tiers must pass the existing 90% capacity test. The common tier is always measured
if native cache verification succeeds; host qualification is recorded separately,
and an unqualified host is marked with its actual IOPS instead of relabelled as
attaining the requested rate. Device/hypervisor caches remain outside full control.

`DiskANN Concurrent Native I/O` runs six desktop CI environments and the Android
and iOS simulators. Its summary validates worker counts, fixed total queries,
live-worker RSS sampling, process-wide rate/depth budgets, storage-read evidence,
frozen budgets/recall and medians. Raw JSON retains each process trial and the
unchanged one-worker source data and provenance.

The [2026-11-05 result tables](results/2026-11-05-concurrent/concurrent-comparison.md)
retain all 378 newly measured query-process trials. `DiskANN Concurrent Results
Audit` additionally requires all eight source measurement jobs to have succeeded
and verifies the complete artifact set without rerunning queries. The baseline
text fingerprint accepts Git's LF/CRLF checkout encodings while independently
requiring exactly equal parsed one-worker data.

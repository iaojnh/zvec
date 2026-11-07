# Buffer Pool release acceptance

The test branch is `test/buffer-pool-acceptance`. The product code comes from
`codex/fix-write-buffer-reservation`; acceptance-only changes stay on the test
branch. A green summary applies only to the scope recorded in
`scripts/buffer_pool_policy.json` and to the exact candidate commit.

## Required stages

1. Build a candidate wheel and compile the complete Linux native test suite.
   Package a fingerprinted native test archive, including
   all build-local transitive shared libraries. Verify relocation before upload
   and set the extracted library directory in each native test job.
2. Run native budget, ownership, eviction, writeback, failure/retry and lifecycle
   regressions in four independent jobs. Run the gate's own negative tests.
3. Run public API mutation, sparse/quantized, FTS/hybrid, result lifetime,
   admission and WAL recovery tests. A skipped FTS compaction witness is an error.
4. Run 100K/200K/400K workloads, four competing collections, and a paired
   128/512/1024 MiB × 1/4/8 worker capacity matrix. All capacity points return
   IDs/scores only, so projection work does not change between budget points.
   Full field/vector projection and writable lifecycle jobs start at 512 MiB.
   Normal cases must finish; an admission error does not turn them into passes.
5. Run each of IVF and DiskANN in one process for at least one hour with a
   fixed corpus, update/delete/upsert/flush, eight readers, periodic optimize,
   and close/reopen. Require 36+ cycles, a stable RSS band and an hourly trend.
6. Run both indexes with a real 1536 MiB cgroup v2 limit and no swap. Require
   the applied numeric limits, complete workload evidence and zero new OOM
   events. Missing cgroup support blocks acceptance.
7. Only after lower stages and platform compatibility pass, run Cohere Medium
   1M with IVF/DiskANN and Buffer/mmap. On each dedicated runner, build and test
   the pinned baseline and candidate sequentially using isolated environments,
   identical inputs/parameters and the same external memory cap (6/8 GiB).
   Delete each temporary index before building the next one. The candidate also
   exercises mutation, unclean exit, writable WAL recovery, restore, optimize,
   reopen, fetch and query at the million-row scale.
8. The summary validates job identities, execution results and mandatory
   artifacts. Missing, cancelled, skipped, incomplete or failing stages cannot
   produce a passing acceptance result. Optional native skips are listed as
   unsupported observations and do not expand the accepted scope.

## Memory evidence

Before every measured workload, three independent processes import the same
program and package, initialize with the same budget/thread settings and open
an empty collection. The imported-process maximum is the baseline; the import
spread is the measurement error. Initialization/open readings are reported but
are not added to the baseline, to avoid counting managed memory twice.

The ceiling is fixed before the workload:

`baseline + managed budget + bounded input batch/query fixture/decoder/stacks + error`

Input conversion uses this interpreter's object sizes and the fixed batch,
dimension and field-size caps. Native input copies fit in the second batch
allowance. The thread allowance covers Python/native request workers and
optimizer/eviction/writeback/Arrow I/O stacks, at 8 MiB each. Cohere's decoder
allowance is twice its largest uncompressed row group; its query fixture is
bounded separately. Managed result copies are not added as outside overhead.
The components, package hash, platform, page size, selected I/O backend,
current RSS and native process high-water mark are retained in JSONL.
Linux also records anonymous/file RSS, swap, thread count and glibc live/free
heap bytes. Python Arrow allocation is recorded when that input reader is
loaded. Every sixth close sample is echoed in the job log for diagnosis.

All RSS samples, including warm-up, must respect the ceiling. The final two
steady windows may grow by at most 8 MiB; hour-long runs also require a fitted
slope at most 8 MiB/hour. There is no forced GC, allocator trim or runtime
allocator tuning. Physical vector/FTS compaction and pressure-induced eviction
must have execution witnesses; calling optimize alone is insufficient.

mmap is a performance reference. Its mapped-file RSS is reported rather than
claimed to obey the Buffer Pool ceiling. The same external cgroup cap applies
to both storage modes and both compared revisions.

## Performance gate

The immutable baseline SHA is in the policy (initially upstream commit
`30d662db4d2467937d843443074b34743bcc9baa`). Changing it is an explicit policy
change, rather than silently comparing against a moving branch.
The Cohere reader uses a 64 KiB input buffer with prebuffering disabled: a
256-row output batch alone otherwise reads a ~1 GiB compressed column chunk.

Each distribution/cache point has three trials, with the median gated:

| Metric | Required candidate result |
| --- | --- |
| QPS | At least 80% of baseline (user-approved) |
| P99 | At most 125% of baseline (user-approved) |
| Physical reads | At most 125% of baseline + 1 MiB measurement floor |
| Recall@10 | At least 0.90 in every trial |

Distributions are uniform unique, 80/20 query-ID hotspots, truncated Zipf
alpha=1 and 90/10 exact repeat. These are reproducible request distributions;
they do not claim a measured centroid/semantic page working set.

Cold-start trials drop OS caches before opening and require physical-read
evidence. Cold QPS and I/O include opening; hot QPS and I/O start after an
untimed warm-up of that exact sequence. P99 measures query calls in both cases;
query-only read bytes are also retained. The runner is dedicated and ephemeral.

## Support boundary

Linux glibc x64 receives the full memory, hour-long and million-row gate.
Linux ARM64, macOS ARM64/x64 and Windows x64/ARM64 receive the 20K IVF/DiskANN
memory lifecycle plus their normal platform suites. musl, Android and iOS
retain their workflow's compile/emulator/simulator compatibility scope.
Physical mobile devices and full large-scale memory acceptance on those other
platforms remain outside this gate. The CI does not certify arbitrary host
allocations, unbounded result sizes or every possible concurrency setting.

Acceptance is complete for this fixed scope only when the final report passes.
Adding a job or pushing the workflow is not evidence that the version passed.

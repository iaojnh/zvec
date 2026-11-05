# DiskANN cross-platform publication benchmark

Base: `iaojnh/zvec:codex/pq-training-zero-copy` at `772c5981`.
The test branch also ports the DiskANN platform gate from `52033441` so Intel
macOS and Windows ARM64 run the actual implementation instead of a stub.

The dedicated **DiskANN Platform Benchmark** workflow runs automatically on this
test branch and can be dispatched with 100,000–1,000,000 SIFT vectors. Its first
run uses 100,000 FP32 128-dimensional vectors, 200 calibration queries, and 1,000
held-out evaluation queries. All jobs receive the same prepared dataset artifact
and SHA-256 manifest. The source is the public
[ANN-Benchmarks SIFT dataset](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py).
Exact top-10 ground truth is recomputed for the selected subset.

## Coverage

- Native CI: Linux x86_64/ARM64, macOS Intel/Apple Silicon, Windows x86_64/ARM64.
- Android x86_64 emulator and iOS ARM64 simulator: separate results, explicitly
  excluded from physical mobile-device publication rows.
- This workflow has no physical Android/iPhone runner. Those publication rows
  remain empty until the same benchmark is run on actual devices.

## Measurement

- Same C++ core index interface, FP32 squared Euclidean distance, MMAP storage,
  no additional vector quantizer, one concurrent query for both indexes.
- HNSW: M=32, ef_construction=200. DiskANN: degree=64, construction list=100,
  PQ chunks=dimension/4 (32 for SIFT).
- Build and query run in separate OS processes. The query process loads only
  query vectors and ground truth; it never loads the base-vector matrix.
- Choose the first budget in 32,48,64,96,128,192,256,384,512,768,1024,1536,2048
  that reaches 0.95 Recall@10 on calibration queries. Warm the held-out query
  set once, then measure it three times. Calibration queries are excluded.
- Report held-out Recall@10, P50/P95/P99, serial search-only QPS, total process
  RSS/Windows working set, RSS delta over the pre-open process, sampled peak
  query RSS, and build wall time. Memory uses decimal MB (1,000,000 bytes).
- Publication rows require held-out recall >=0.93 and pairwise recall gap <=0.02.
  If either condition fails, retain the raw measurements, fail that job, and
  leave its publication row empty. Do not imply perfectly identical recall.
- QPS is the inverse aggregate native search time at concurrency 1; it is not a
  saturated throughput measurement or full Collection/Python API performance.
- Query-workload warmup does not guarantee OS page-cache hits. Direct readers
  may continue performing storage I/O; Apple no-cache hints can still hit
  existing pages. CI runner virtualization, CPU, memory, storage, and image
  information are recorded; these are not consumer-device guarantees or an
  operating-system performance ranking. RSS excludes kernel file cache that
  is not mapped into the query process; RSS savings are not total-memory savings.

## Query-cache controls

CI also enables `--cache-comparison`. Each platform builds each index once and
reuses its calibrated budget without performing any calibration in the replay
workers. Index and query SHA-256 hashes and identical recall are checked.
Three fresh-process trials run for each index and state:

- **First pass:** attempt a cache reset before opening the index, omit query
  warmup, and time one held-out pass. This warms naturally during the pass; it
  does not make every query cold. Linux uses a host cache reset; Apple uses
  host `purge`; Android attempts a guest-only reset. Windows has no verified
  reset here and is explicitly labeled as an uncontrolled first-pass cache state.
- **Steady:** search the held-out workload once, then time three passes.

Cache resets are restricted to ephemeral GitHub-hosted VMs, never a developer's
computer or physical phone. Record command success separately from evidence of
actual I/O. The benchmark records index-open, warmup and timed read bytes, open
time, timed CPU time and major page faults. Linux/macOS counters are OS storage
accounting, Windows bytes are logical transfers, and iOS has no disk-byte counter
used here. Guest/hypervisor caches remain separate concerns. The comparison
table reports medians across trials, not pooled latency percentiles.

Windows ARM64 uses the unique verified blank local NVMe scratch disk when
available; the provisioning metadata and actual data path are retained.

## Android direct-I/O validation

Android keeps synchronous pread, but now attempts an independent O_DIRECT
descriptor for graph/full-vector reads in both open paths. A captured buffered
descriptor refers to the same inode, leaves metadata-handle flags unchanged,
and remains available for explicit open/read compatibility fallback. Direct
read EINVAL/EOPNOTSUPP/ENOSYS triggers a one-time warning and buffered retry;
EOF and actual storage failures remain errors. Descriptors are retained until
close so in-flight concurrent readers remain valid during fallback.
When available, per-file STATX_DIOALIGN also detects a file for which the kernel
accepts O_DIRECT but reports no direct-I/O support. Unknown capabilities retain
the direct attempt. The syscall is gated to Android 11+ to preserve older
Android seccomp compatibility and avoid a newer bionic symbol dependency.

The **DiskANN Android Direct I/O** workflow builds ARM64 and x86_64. The x86_64
emulator runs deterministic open/read failure, pathname replacement, source
flags, EOF/storage error, and concurrency regressions, followed by a production
reader probe and the 100k-vector query/cache benchmark. The standalone probe
uses a nonzero 64 MiB file and checks all bytes of 4096 aligned 4 KiB reads per
case. Warm/cold buffered controls establish working storage accounting and
mincore file-cache residency before assessing the production reader.

Results distinguish **verified direct reads**, **explicit buffered fallback**,
and **O_DIRECT requested but bypass not verified**. A requested flag alone is
never a bypass claim. Verification is confined to the Android guest page cache
and storage accounting, not the emulator host or physical phone flash.
An additional unencrypted-directory positive control and query run separates
filesystem support from encrypted userdata fallback. Preparing that directory
uses root only inside the verified ephemeral CI emulator; no security policy
is disabled and ordinary app files are not moved to an unencrypted location.

## Outputs

Each platform uploads build/query logs, raw JSON, and `result.json`.
The `diskann-platform-summary` artifact contains `publication_table.md`,
`comparison.md` (including both algorithms, simulators, and test conditions),
and `results.json`. Cache controls additionally produce `cache-comparison.md`
and full `cache-comparison.json`, plus per-trial logs/JSON in each platform
artifact. Missing/failed/non-comparable platforms keep blank cells.
The benchmark GTest skips unless `ZVEC_BENCH_MODE` is set, so normal unit-test
runs do not perform this benchmark.

## Local execution

```sh
python -m pip install numpy h5py
python scripts/prepare_diskann_platform_data.py --output benchmark-data --count 100000
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build-benchmark --target diskann_platform_benchmark_test --parallel 3
python scripts/benchmark_diskann_platform.py \
  --binary build-benchmark/bin/diskann_platform_benchmark_test \
  --data benchmark-data --output benchmark-results --label local
```

For a quick local harness smoke test only, prepare data with `--synthetic
--count 512 --queries 220`. Synthetic smoke measurements are not publication data.

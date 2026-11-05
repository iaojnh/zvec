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
- Warm caches can include operating-system file cache. Results must not be
  described as cold-disk timings. CI runner virtualization, CPU, memory, storage,
  and image information are recorded; these are not consumer-device guarantees.

## Outputs

Each platform uploads build/query logs, raw JSON, and `result.json`.
The `diskann-platform-summary` artifact contains `publication_table.md`,
`comparison.md` (including both algorithms, simulators, and test conditions),
and `results.json`. Missing/failed/non-comparable platforms keep blank cells.
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

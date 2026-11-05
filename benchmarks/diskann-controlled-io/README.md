# Controlled DiskANN storage costs

This opt-in experiment removes real disk latency from graph/full-vector reads.
It does **not** make different CPUs, memory systems or operating systems into
identical physical hardware. Use a fixed physical machine with native multiboot
for a hardware-controlled OS comparison. GitHub-hosted results cannot replace it.

The benchmark preloads the same canonical index file into RAM before timing,
then routes DiskANN's actual read/submit/completion calls through a discrete-event
model: no file cache, 200 microseconds per request, at most eight requests in
service, and one shared 1,048,576,000 byte/second transfer channel. Transfer time
is rounded up to integer nanoseconds. A request occupies its service slot until
its transfer plus fixed latency completes. All requests and bytes are charged,
including duplicate sector reads. Dependencies between search batches remain.
The model resets at each serial query; it does not model saturated multi-query
throughput, CPU/I/O overlap, OS completion overhead, or storage-device firmware.
No OS sleep or timer controls the modeled costs.

The CPU still executes the real PQ/distance/graph-search code. Report actual
memory-backed wall time and process CPU time separately from synthetic I/O-only
cost. Do not add those times and call the result native end-to-end performance.
The full backing image is resident, so this mode's RSS is **not** a DiskANN
product-memory measurement. Use the separate native benchmark for RSS, SSD QPS,
and platform-reader/cache validation.

The new reader hook is compiled out unless
`-DDISKANN_CONTROLLED_IO_BENCHMARK=ON`. Ordinary builds reject attempts to enable
the mode at runtime. Existing platform-reader correctness and cache CI remain
independent of this experiment.

## Reproduce

Build the `diskann_platform_benchmark_test` and `diskann_controlled_io_test`
targets in Release mode with the option above. Run the model regression tests.
Prepare the standard SIFT data with `scripts/prepare_diskann_platform_data.py`.

On one canonical host, train both indexes and freeze their search budgets:

```sh
python scripts/benchmark_diskann_controlled_io.py \
  --binary build/bin/diskann_platform_benchmark_test \
  --data benchmark-data --output canonical-results \
  --label canonical --prepare-fixture
```

Copy the entire `benchmark-data` directory to every target. It includes the
trained indexes, exact ground truth, queries, and `controlled-fixture.json`.
Do not rebuild indexes on the replay hosts:

```sh
python scripts/benchmark_diskann_controlled_io.py \
  --binary build/bin/diskann_platform_benchmark_test \
  --data benchmark-data --output replay-results --label local
```

Parameters `--latency-ns`, `--queue-depth`, and `--bytes-per-second` select
explicit alternative models. Keep the model settings identical across an OS
comparison. The same-index manifest is checked before/after each run, transfers
are hashed on the mobile targets, and replay fails on a changed search budget,
recall, or inactive reader. The eight-environment workflow prepares the fixture
once, then replays it on six native OS/architecture runners and two explicitly
labeled mobile simulators.

Outputs: full `controlled-results.json`, per-process JSON/logs, and
`controlled-comparison.md` with read count, bytes, batches, modeled I/O-only
P95/QPS, and separately labeled host execution timing. Keep synthetic smoke
data out of release performance claims.

# OpenViking native local Flat vs zvec Flat

Tested on 2026-09-01 using an Apple M3 Pro (arm64), macOS 26.6.2, and
Python 3.14.6.

- OpenViking: 0.4.17.1
- zvec: 0.7.1.dev16, checkout `7a682cf87b9202f67dd2deba76f8e695ea6dd2ab`
- Dataset: 50,000 normalized FP32 vectors, dimension 384
- Workload: 500 queries, top 10, 50 warmups, 3 measured repeats
- Query generation: noisy copies of base vectors, noise 0.03, seed 20260901
- Ground truth: exact NumPy FP32 dot-product top 10
- Isolation: every backend/mode/layout ran in a separate process

The table is the arithmetic mean of two complete, independently rebuilt runs.

| mode | backend | layout | Recall@10 | QPS | p50 ms | p95 ms | p99 ms | load + finalize s | warm RSS MiB | disk MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 | OpenViking | contiguous | 0.9812 | 1084.3 | 0.886 | 1.091 | 1.715 | 5.39 | 702.0 | 97.0 |
| INT8 | zvec | contiguous | 0.9832 | 103.9 | 9.566 | 10.402 | 11.003 | 1.19 | 226.2 | 99.4 |
| INT8 | zvec | default | 0.9832 | 99.4 | 9.910 | 11.041 | 11.779 | 1.21 | 205.0 | 99.4 |
| FP32 | OpenViking | contiguous | 1.0000 | 575.1 | 1.667 | 2.199 | 2.865 | 5.69 | 758.8 | 151.8 |
| FP32 | zvec | contiguous | 1.0000 | 478.5 | 2.013 | 2.491 | 4.369 | 1.18 | 327.2 | 78.3 |
| FP32 | zvec | default | 1.0000 | 352.0 | 2.783 | 3.246 | 3.916 | 1.15 | 253.7 | 78.3 |

## Conclusions

1. OpenViking's native local index in this test is exhaustive Flat. Its actual
   persisted metadata was `IndexType=flat`, with IP distance and either INT8 or
   float storage.
2. The like-for-like FP32 contiguous comparison is close: OpenViking delivered
   1.20x zvec's QPS and 17% lower p50 latency. Both had exact Recall@10.
3. zvec's contiguous layout mattered for FP32: it delivered 1.36x its default
   layout's QPS and 28% lower p50 latency.
4. INT8 was not close on this Apple ARM host: OpenViking delivered 10.4x the QPS
   of contiguous zvec, with almost the same recall (zvec was 0.002 higher in
   absolute Recall@10). zvec contiguous and default were nearly identical here,
   so the gap is not explained by the layout setting.
5. zvec loaded and finalized the data about 4.5x faster for INT8 and 4.8x faster
   for FP32.

## Interpretation limits

- INT8 is a product-level comparison, not the same quantizer implementation.
  OpenViking and zvec use different per-vector quantization formulas.
- Source inspection suggests the large INT8 result is platform/path-specific:
  zvec's inner-product INT8 batch path selects its fallback template, while its
  accelerated integer kernels are primarily wired to other metrics/architectures.
  This needs a profiler and a Linux x86 rerun before treating the 10x ratio as a
  general result.
- RSS is whole-process resident memory after warmup, including Python, loaded
  libraries, generated data, and memory-mapped pages. Disk is the complete local
  collection, not only vector payload. Neither should be read as index-only size.
- The benchmark bypasses embeddings, HTTP, filters, reranking, and document
  parsing. It compares the persistent collection APIs and their Flat search path.

The individual reports are in `results/summary.md` and
`results/repeat2/summary.md`; raw measurements are in the adjacent JSON files.

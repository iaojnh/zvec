# Why OpenViking INT8 Flat is much faster on Apple ARM

This analysis applies to the measured Apple M3 Pro arm64 environment. The
observed end-to-end p50 values were 0.886 ms for OpenViking and 9.566 ms for
contiguous zvec, a 10.8x latency ratio.

## Primary cause: integer versus float reduction

The installed OpenViking 0.4.17.1 binary accumulates an INT8 dot product in
`int32_t`, then converts the final sum to float once. Apple Clang vectorizes the
loop into NEON widening multiply-accumulate instructions (`sshll` + `smlal`),
keeps several independent vector accumulators, horizontally reduces them, and
emits one `scvtf` after the loop.

zvec's Apple ARM fallback instantiates the generic scalar template. Its source
uses a float accumulator:

```cpp
float sum = 0.0;
for (size_t i = 0; i < dim; ++i) {
  sum += static_cast<float>(m[i] * q[i]);
}
```

The compiler vectorizes the INT8 multiplications, but preserving the float
reduction order produces repeated `scvtf` conversions and a long serial chain
of scalar `fadd` instructions. For every 32 dimensions, the generated code
extracts and converts all 32 products to float and adds them one by one to the
same accumulator. This is throughput- and dependency-chain-heavy.

At dimension 384, an exact signed INT8 dot product is at most 6,193,536, so an
`int32_t` accumulator is safe for this benchmark.

## Direct kernel experiment

`kernel_microbench.cpp` reproduces the two accumulation forms using the same
50,000 x 384 workload and the actual record strides. It was built with Apple
Clang `-O3` and run four times. Median results:

| kernel | record stride | ms per 50k-vector scan |
|---|---:|---:|
| OpenViking-style INT32 accumulation | 400 bytes | 0.388 |
| INT32 accumulation with zvec stride | 448 bytes | 0.420 |
| zvec-style float accumulation | 448 bytes | 9.140 |

- Float versus INT32 at the same stride: about 22x (19.9x to 22.6x).
- The median 448-byte versus 400-byte stride penalty: about 7%.
- The 9.14 ms float-kernel result is almost the same magnitude as zvec's 9.57
  ms end-to-end p50. This strongly identifies the distance loop as the dominant
  cost rather than Python or Top-K processing.

The end-to-end ratio is smaller than the isolated roughly 22x kernel ratio because
OpenViking's 0.886 ms also includes query quantization, 50,000 indirect distance
calls, scale correction, label reads, Top-K maintenance, and Python result
conversion. Once the dot product becomes fast, those fixed costs matter.

## Why zvec selects this path

For INT8 inner product, `QuantizedIntegerMetric::batch_distance()` returns
`BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, int8_t, 12, 2>`.
The wrapper has optimized dispatch branches for cosine and squared L2, but not
for `MinusInnerProduct`, so IP falls through to `_ComputeBatch()`. That function
loops over vectors and calls the one-vector distance implementation.

The one-vector INT8 dispatcher has AVX2 and SSE implementations, but no ARM
NEON implementation. On Apple ARM it therefore calls `InnerProductInt8Scalar`,
which is the float-accumulator loop above. The existing one-to-many INT8 batch
implementation also only dispatches to AVX512 VNNI or AVX2 before falling back;
it has no ARM specialization.

This makes the result partly platform-specific. The same IP path also misses
the one-to-many batch kernel on x86, but its per-vector AVX2/SSE fallback should
make the penalty much smaller than on Apple ARM.

## Secondary contributors

### Quantization format

OpenViking uses symmetric per-vector quantization and stores 384 INT8 values
plus one 4-byte scale. Its score correction is one integer-to-float conversion
and two scale multiplications.

zvec uses asymmetric per-record min/max quantization. It stores 20 metadata
bytes after the 384 INT8 values: scale, bias, sum, squared sum, and integer sum.
It then evaluates several correction terms for every vector. This gives zvec
slightly higher recall in the benchmark (0.9832 versus 0.9812), but adds work.

### Record layout

- OpenViking scans tightly packed 400-byte records: 388-byte encoded vector,
  8-byte label, and 4-byte logical offset.
- zvec's contiguous path has a 404-byte quantized vector and rounds every record
  to a 64-byte boundary, giving a 448-byte stride, while keys are kept in a
  separate array.

The direct experiment measured this as only a roughly 7% median penalty, so it
cannot explain the 10x result.

### Scan and Top-K structure

OpenViking performs one direct record loop and only modifies its priority queue
when a candidate enters Top-K. zvec first builds batches of vector/key pointers,
computes distances, then loops over the results and calls its document heap for
every vector. These costs will become visible after fixing the distance kernel,
but are currently hidden under the 9 ms dot-product loop.

### Python and collection overhead

Both benchmarks convert queries to Python lists and call persistent collection
APIs. In FP32 mode both engines have ARM NEON kernels, and the measured gap was
only about 20%. That is additional evidence that wrapper/API overhead is not the
source of the INT8 10x gap.

## Likely remediation order for zvec

1. Add an ARM NEON INT8 inner-product kernel that accumulates in 32-bit integer
   lanes and converts once after horizontal reduction.
2. Wire `MinusInnerProduct` through the existing one-to-many batch path instead
   of `_ComputeBatch()`; add an ARM specialization for batch sizes 12/32.
3. Avoid 64-byte padding per 404-byte code, or use a packed scan representation
   separate from mutable record storage.
4. Reduce per-candidate pointer-vector and document-heap construction in the
   unfiltered Flat scan.
5. Re-run the full benchmark, profiler, and recall checks on Apple ARM and Linux
   x86 (AVX2 and AVX512 VNNI) separately.

The first item should recover most of the Apple ARM loss. The kernel experiment
suggests that layout-only work cannot do so.

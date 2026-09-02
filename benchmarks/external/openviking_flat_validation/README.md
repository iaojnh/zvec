# OpenViking local Flat vs zvec Flat

This benchmark compares the persistent Python collection APIs of OpenViking's
native local Flat index and zvec's Flat index. It deliberately bypasses
embedding, HTTP, filtering, reranking, and document parsing.

Two modes are measured:

- `int8`: both indexes receive normalized FP32 vectors and use IP plus INT8
  quantization. This compares product behavior; the quantizers are not
  numerically identical.
- `fp32`: both indexes receive normalized FP32 vectors and use unquantized IP.
  This is the like-for-like exact Flat comparison.

The generated queries are noisy copies of base vectors. Ground truth is a
NumPy FP32 exhaustive dot-product search over the same normalized vectors.

See `RESULTS.md` for the two-run benchmark summary and `ANALYSIS.md` for the
Apple ARM INT8 kernel analysis and targeted microbenchmark.

## Run

```bash
python benchmark.py matrix \
  --n 20000 --dim 256 --queries 500 --topk 10 \
  --repeats 3 --warmup 50 --batch-size 1000
```

The command runs each backend in a separate process and writes JSON plus a
Markdown summary under `results/`.

Use `--zvec-layouts contiguous,default` to report both zvec's layout-aligned
contiguous mode and its default non-contiguous mode.

## Apple ARM INT8 kernel diagnostic

```bash
clang++ -O3 -std=c++17 kernel_microbench.cpp -o .workspace/kernel_microbench
.workspace/kernel_microbench
```

This isolates the float-accumulator loop used by zvec's ARM fallback from the
integer-accumulator loop emitted by the installed OpenViking binary.

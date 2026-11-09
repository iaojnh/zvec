# Cohere 1M IVF release profile

This profile checks same-device build, sustained writes with queries, persistence
and IVF memory efficiency. It reuses the existing CI's versioned Cohere Medium
1M data, public neighbor ground truth and IVF parameters. All database operations
use public Zvec APIs, including `CollectionOption(max_buffer_size=...)`.

## Workload and memory sweep

The corpus has 1,000,000 FP32 vectors of dimension 768 (2929.7 MiB of raw vectors).
The profile uses COSINE, 1024 lists, 5 training iterations and nprobe=128, four
query workers and one optimization worker. Two hundred real query vectors have
public top-10 neighbors. The input is normalized for cosine; a streaming exact
oracle independently cross-checks the public neighbors before any ANN results
can pass. Require at least 99% agreement with the public ground truth.

Buffer Pool soft targets are **512, 1024, 1536, 2048 and 4096 MiB**. Every point
runs the full lifecycle inside an external allowance of its target plus 256 MiB,
with swap disabled. That extra room is a tested workload condition, not a general
bound on RSS overshoot. RSS exceeding the soft target is recorded. No universal
overshoot amplitude, 2-second recovery or 1% exceedance SLA is asserted.

The mmap reference gets **12288 MiB** as both soft target and external allowance
in every stage. Its measured query RSS must stay below 80% of that target, and
query trials must show no external-limit pressure. Actual whole-process RSS,
not the configured allowance, is the denominator in QPS/RSS.

Each point and its mmap reference use the same CI runner, wheel, input and
parameters. Different points run in parallel on separate runner instances;
the normalized ratio uses each point's paired reference. Raw QPS across points
also reflects runner variation. Environment details accompany every point.

The write buffer is 1 MiB and batches contain at most 256 documents. After build,
the mixed stage inserts 2048 new IDs and updates those IDs once, using 16 flushed
batches paced across at least 60 seconds with a concurrent reader. Optimization
runs while the reader remains active. Exact truth incorporates the new vectors;
the final live count is 1,002,048. A fresh process verifies queries, fields,
vectors and last-update payloads. Build also tests reopen/query before its own
process exits, covering retained build-time heap at the transition.
After each flushed batch, the writer searches the new vector and requires its
ID at top-1 before optimization, while the background reader remains active.
The fresh-process check repeats that search after reopen. Thus mixed validation
can briefly have two query callers; `readers=1` is the background worker count.

## Prepare input and run

The public training Parquet file has large column chunks; its Arrow decoder can
need several GiB independently of Zvec. A separate fixture-preparation process
downloads the same ETag/size-pinned files as existing CI, preserves IDs, normalizes
vectors and emits checksummed binary input. Preparation runs on the same device
before qualification and needs additional memory and disk. It is not included
in the Zvec process memory envelope. The measured build reads small batches from
those files without retaining an input mmap. Applications can likewise stream
bounded batches from their own source.

Use Ubuntu 24.04 x86_64/Python 3.10, the matching candidate wheel and these files:

```bash
python -m pip install /path/to/candidate.whl pyarrow
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python ivf_cohere_input.py --profile ivf_cohere_profile.json \
  --source ./cohere-parquet --output ./cohere-input --download
mkdir results
for stage in build mixed verify; do
  python ivf_memory_profile.py --profile ivf_cohere_profile.json \
    --input-dir ./cohere-input --root ./cohere-run/buffer --storage buffer \
    --stage "$stage" --result "results/buffer-$stage.json"
done
for trial in 1 2 3; do
  python ivf_memory_profile.py --profile ivf_cohere_profile.json \
    --input-dir ./cohere-input --root ./cohere-run/buffer --storage buffer \
    --stage query --trial "$trial" --result "results/buffer-query-$trial.json"
done
```

The candidate JSON defaults to 2048 MiB. A qualified exported JSON can directly
replace it in the example's `--profile` argument. Use a fresh root/result directory
for each storage mode or budget; result files are created exclusively. Direct
execution checks functionality; CI adds independent RSS sampling and OS limits.
For the reference, repeat with `--storage mmap` and separate root/result paths.
In a repository checkout, select another declared point with:

```bash
python .github/scripts/buffer_pool_ivf_sweep.py configure \
  --profile examples/python/ivf_cohere_profile.json --memory-limit-mb 1024 \
  --output ./profile-1024.json
```

## Acceptance and exported artifacts

The existing Buffer Pool Acceptance workflow runs the five pairs and a separate
400K random-vector memory stress test. It reuses candidate building, checksums,
native regression binaries, the RSS observer and cgroup wrapper. The input manifest
and environment information are included. Prepared raw input is not uploaded as
part of the small evidence artifact; the preparation command reproduces it.
Lifecycle RSS peaks are sampled observations and include their sampling-gap
statistics; the cgroup cap covers intervals between samples. The query projection
uses `output_fields=[]` and `include_vector=False`; full field/vector correctness
is checked separately by fetching probes before and after writes.

For uniform, hot-80/20, Zipf and repeat-90/10 request distributions, run three fresh
process trials. Each measures at least 200 queries and three seconds after 16
warmup queries. Every measured query checks recall; each process also checks the
entire fixed query set. Require recall at least 0.9 and a difference from mmap
of at most 0.01. Trial order alternates storage modes.

An independent process samples whole-process RSS every 100 ms. RSS is integrated
over the exact measured query window, without baseline subtraction or a peak
divisor. Excessive gaps or missing bracketing samples fail acceptance. A point
qualifies only if its median per-trial **QPS / average RSS MiB is strictly higher
than ample-memory mmap in every declared distribution**, all lifecycle checks
pass, the source/wheel/input match, and no OOM or swap occurs. Raw QPS and p99
describe the memory/performance tradeoff; they are not separate SLO promises.

Each `buffer-pool-ivf-profile-<budget>m` artifact contains selected controls, runnable
examples, raw stage results, RSS samples, cgroup observations and readable reports.
Failed points remain visible and cannot become recommended configurations.
`buffer-pool-ivf-pressure-512m` contains the independent random-data pressure result.

The `buffer-pool-ivf-sweep` artifact summarizes every requested point. Only a
complete sweep, a passing random pressure test and at least one qualified Cohere
point produce `ivf-profile-qualified.json`, selecting the lowest qualified budget.
The configuration retains runtime controls at the top level and adds a receipt
with source commit, wheel hash, dataset manifest hash, environment and comparisons.
It is runnable with the public example. Changes to data, projection or parameters
require new qualification; these measurements do not establish all-workload or
all-hardware performance guarantees.

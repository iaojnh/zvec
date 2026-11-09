# IVF memory profile

This example builds an IVF index, performs sustained writes concurrently with
queries, closes and reopens the collection, and measures query throughput.
It uses only public Zvec APIs, including `CollectionOption(max_buffer_size=...)`.
Use the wheel produced by the same candidate commit as these files.

## Candidate configuration

`ivf_memory_profile.json` defines the entire experiment. The initial candidate
uses Linux glibc x86_64, 400,000 FP32 vectors of dimension 768, L2 IVF with
32 lists, 5 training iterations, and nprobe=32. The corpus is synthetic normal
data with a fixed seed; 16 fixed queries have independently computed exact
top-10 ground truth. These results do not establish performance on other data.

Buffer Pool runs every stage under a 768 MiB cgroup limit with swap disabled,
using a 512 MiB managed budget/RSS soft target. Both storage modes use that same
768 MiB external limit for the actual query comparison. The mmap query reference
uses a 768 MiB soft target, following the existing CI convention that file-backed
RSS must not cause artificial admission failures at the Buffer Pool target.
All values are recorded; actual whole-process RSS is the metric denominator.

The first trial at commit `85830989` passed all Buffer Pool lifecycle stages,
but mmap training returned `Not enough space` under the 768 MiB allowance.
Consequently mmap baseline preparation (build/mixed/verify) explicitly uses a
separate 4096 MiB external limit and soft target. It then exits; query trials
start fresh at the shared 768 MiB external limit. This allowance prepares a
performance reference; it does not claim mmap can build within the Buffer Pool
deployment envelope. The initial failed run remains available as evidence.

All stages use a 1 MiB write buffer, batch size 128, one query worker and four
optimization workers. The mixed stage inserts 1,536 additional vectors, then
updates those same IDs once while a reader runs. Its 24 batches are paced across
at least 60 seconds. It flushes each batch, optimizes, checks the resulting count
and verifies the last update. Identical 3,072 write operations keep both live rows
and obsolete vector versions comparable across modes. Slower operations may
exceed 60 seconds.
The build stage also closes, reopens and queries before its process exits, so
retained build-time heap memory is included in the transition check.

## Run the public example

Install the candidate wheel (it installs NumPy), then use a new output directory:

```bash
python -m pip install /path/to/candidate.whl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir ivf-results
for stage in build mixed verify; do
  python ivf_memory_profile.py --profile ivf_memory_profile.json \
    --root ./ivf-run/buffer --storage buffer --stage "$stage" \
    --result "ivf-results/buffer-$stage.json"
done
for trial in 1 2 3; do
  python ivf_memory_profile.py --profile ivf_memory_profile.json \
    --root ./ivf-run/buffer --storage buffer --stage query --trial "$trial" \
    --result "ivf-results/buffer-query-$trial.json"
done
```

Each command starts a fresh process with the same configuration. Output files
are created exclusively to avoid mixing old and new results. Repeat with
`--storage mmap`, a separate root and result paths to reproduce the reference.
Direct example execution checks functionality; it does not apply OS limits or
generate a qualified deployment profile. The CI below supplies those controls.

## CI acceptance

The existing Buffer Pool Acceptance workflow has an IVF-only dispatch option.
It reuses its candidate wheel build, SHA-256 checks, independent RSS observer
and cgroup wrapper. It executes this exact public example under the configured
memory limit and disables swap. Fresh OS caches before each service ensure
file-cache charges belong to the measured cgroup. Query trials alternate the
storage order and use identical warmup, query sequences and recall parameters.

For each of uniform, hot-80/20, Zipf and repeat-90/10 query distributions:

* Run three trials, each measuring at least 16 queries and three seconds after
  four warmup queries. Each fresh process also checks all 16 oracle queries
  before its cases; every measured query checks its own recall again.
* Require recall at least 0.9, and Buffer Pool/mmap recall difference at most 0.01.
* Compute whole-process RSS from a separate process at 100 ms intervals. Integrate
  RSS over the exact query measurement window; do not subtract a baseline or
  divide by peak RSS. Missing bracketing samples or excessive gaps fail acceptance.
* Require the median per-trial `QPS / average RSS MiB` to be **strictly greater**
  for Buffer Pool than for mmap in every declared distribution.
* Keep the existing guardrails: QPS at least 80% of mmap and p99 at most 125%.
* Require successful build, concurrent writes/readers and fresh-process
  verification for this same profile, wheel and source commit, with no OOM
  events or swap use. Buffer Pool lifecycle stages use its deployment envelope;
  mmap preparation uses its separately declared allowance above.

RSS exceeding 512 MiB is recorded rather than judged against a universal
overshoot amplitude/duration promise. The complete workload must still succeed
inside the declared external limit.

The `buffer-pool-ivf-profile` artifact always contains raw results, independent
RSS samples, cgroup observations, this example, its configuration and
`ivf-profile-report.json` and a readable `ivf-profile-report.md`.
Only a fully passing run generates
`ivf-profile-qualified.json`. A failed or incomplete run is not a recommended
deployment configuration. Changing the corpus, parameters or projection
requires a new qualification run.
The qualified JSON keeps controls at the top level and adds a `qualification`
receipt. It can directly replace `ivf_memory_profile.json` in the example's
`--profile` argument.

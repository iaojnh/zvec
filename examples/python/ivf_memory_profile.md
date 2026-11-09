# IVF memory profile

This example builds an IVF index, performs sustained writes concurrently with
queries, closes and reopens the collection, and measures query throughput.
It uses only public Zvec APIs, including `CollectionOption(max_buffer_size=...)`.
Use the wheel produced by the same candidate commit as these files.

## Memory sweep and reference

`ivf_memory_profile.json` declares Buffer Pool budgets of **256, 512, 768, 1024
and 1536 MiB**. Every point runs the complete lifecycle. Its external cgroup
allowance is the selected budget plus 256 MiB, with swap disabled. This extra
allowance is part of the tested workload configuration, not a universal RSS
overshoot guarantee. Budget, actual RSS, failures and throughput are all reported.

The mmap reference gets a generous **8192 MiB** soft target and external allowance
in every stage, including queries. It is not restricted to the Buffer Pool
budget. Qualification also checks that mmap peak RSS is below 80% of its target
and that its cgroup did not reach its external limit. If those checks fail, the
reference needs more room before it is a valid ample-memory baseline.

Each Buffer Pool point and its mmap reference run on the same CI runner with
the same wheel, corpus, write operations, query settings and recall requirement.
The five pairs run in parallel. Within a pair, query trials alternate storage
order. The first run at commit `85830989` demonstrated why the reference needs
its own allowance: Buffer Pool passed all lifecycle stages at 512 MiB, whereas
mmap training returned `Not enough space` under a shared 768 MiB external cap.

## Fixed workload

`ivf_memory_profile.json` defines the entire experiment. The initial candidate
uses Linux glibc x86_64, 400,000 FP32 vectors of dimension 768, L2 IVF with
32 lists, 5 training iterations, and nprobe=32. The corpus is synthetic normal
data with a fixed seed; 16 fixed queries have independently computed exact
top-10 ground truth. These results do not establish performance on other data.
The largest budget is an in-memory control; smaller budgets exercise corpora
larger than the managed budget.

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

Install the candidate wheel (it installs NumPy), then use a new output directory.
The candidate JSON defaults to the 512 MiB point; an exported qualified JSON can
replace it directly in the `--profile` argument.

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

In a repository checkout, select another declared point with:

```bash
python .github/scripts/buffer_pool_ivf_sweep.py configure \
  --profile examples/python/ivf_memory_profile.json --memory-limit-mb 768 \
  --output ./profile-768.json
```

## CI acceptance and exported configuration

The existing Buffer Pool Acceptance workflow has an IVF-only dispatch option.
It reuses its candidate wheel build, SHA-256 checks, independent RSS observer
and cgroup wrapper. It executes this exact public example under the configured
memory limit and disables swap. Fresh OS caches before each service ensure
file-cache charges belong to the measured cgroup. Both modes use identical
warmup, query sequences and recall parameters, with their separately declared
memory allowances.

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
* Report raw QPS and p99 to show the memory/performance tradeoff. They are not
  separate throughput or latency promises.
* Require successful build, concurrent writes/readers and fresh-process
  verification for this same profile, wheel and source commit, with no OOM
  events or swap use inside the declared per-mode external limits.

RSS exceeding the managed budget is recorded rather than judged against a
universal overshoot amplitude/duration promise. The complete Buffer Pool workload
must still succeed inside that point's declared external allowance.

Each `buffer-pool-ivf-profile-<budget>m` artifact contains raw results, independent
RSS samples, cgroup observations, the runnable example, selected configuration
and readable JSON/Markdown reports. A failed point remains visible; it does not
cancel other points. Only a fully passing point exports its own
`ivf-profile-qualified.json`.

The `buffer-pool-ivf-sweep` artifact reports every declared point. Only when the
whole sweep has reported and at least one point qualifies does it export the
lowest qualified budget as `ivf-profile-qualified.json`. It regenerates that
receipt from validated raw evidence. Missing points or mixed candidate versions
prevent aggregate qualification. This JSON keeps runtime controls at the top
level and adds a `qualification` receipt, so it is directly runnable with the
public example. Changing the corpus, projection or parameters requires rerunning
qualification.

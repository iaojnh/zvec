# Random-vector IVF memory pressure test

The release performance profile uses [Cohere 1M](ivf_cohere_profile.md).
This 400K random-vector example remains a separate lifecycle pressure test.

The JSON selects 400,000 FP32 vectors of dimension 768, L2 IVF with 32 lists,
5 training iterations and nprobe=32. Every query scans all lists. This is a
deliberately heavy memory test; request skew does not create a smaller posting
working set. Its optional query measurements do not qualify the release profile.

The tested pressure point uses a 512 MiB Zvec soft target, 768 MiB external cgroup
allowance and no swap. The corpus contains 1172 MiB of raw vectors. A 1 MiB write
buffer, batches of 128, one query worker and four optimization workers bound
concurrency and input sizes. Build verifies reopen/query in the same process;
mixed inserts 1536 vectors and updates them once while queries run, paced across
at least 60 seconds, then optimizes. A fresh process checks all oracle queries,
fields and vectors. A passing stress test exports a pressure report only.

Use the candidate wheel from the same commit, then run from a fresh directory:

```bash
python -m pip install /path/to/candidate.whl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir results
for stage in build mixed verify; do
  python ivf_memory_profile.py --profile ivf_memory_profile.json \
    --root ./random-run --storage buffer --stage "$stage" \
    --result "results/random-$stage.json"
done
```

Direct execution checks functionality. The existing Buffer Pool Acceptance CI
adds independent RSS observation and the external limit. The external allowance
is specific to this workload, not a universal RSS overshoot or Delta guarantee.
Historical random-data sweeps are diagnostic evidence; performance acceptance
and configuration selection now use the Cohere profile.

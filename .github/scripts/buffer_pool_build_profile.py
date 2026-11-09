"""Run the unchanged public-API build workload with independent observation."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import buffer_pool_build_memory as workload
from buffer_pool_build_resources import cgroup_path, elf_identity, key_values, sha256_file
import zvec

def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def configure_probe(manifest, output, budget):
    extension = next(Path(zvec.__file__).parent.glob('_zvec*.so')).resolve()
    assert elf_identity(extension) == manifest['elf'], 'measured ELF differs from frozen build'
    starts = []
    maps = Path('/proc/self/maps').read_text().splitlines()
    for line in maps:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(extension) and int(fields[2], 16) == 0:
            starts.append(int(fields[0].split('-')[0], 16))
    assert len(starts) == 1, 'ambiguous extension load base'
    pointer = ctypes.c_void_p.from_address(starts[0] + manifest['singleton_vaddr']).value
    assert pointer and pointer % 8 == 0, 'memory pool has not been initialized'
    names = list(manifest['fields'])
    offsets = (ctypes.c_size_t * len(names))(*(manifest['fields'][n] for n in names))
    probe = ctypes.CDLL(None)
    probe.zvec_build_probe_start.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                                            ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.c_size_t, ctypes.c_char_p]
    probe.zvec_build_probe_start.restype = ctypes.c_int
    probe.zvec_build_probe_stop.argtypes = []
    probe.zvec_build_probe_stop.restype = None
    shared = (output / 'budget.shared').resolve()
    result = probe.zvec_build_probe_start(pointer, manifest['capacity_cv_offset'],
                                          offsets, len(names), os.fsencode(shared))
    assert result == 0, f'budget observer failed: errno {result}'
    state = dict(pid=os.getpid(), field_names=names, shared_file=str(shared),
                 phase_file=str((output / 'build.jsonl').resolve()), budget_mib=budget,
                 extension=str(extension), extension_sha256=sha256_file(extension),
                 native_sampler_stack_bytes=128 * 1024, diagnostic_shared_bytes=4096,
                 budget_wait_semantics='thread time inside this pool capacity condition variable; includes mutex reacquisition; overlaps across threads')
    dump(output / 'budget-state.json', state)
    return probe

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection')
    parser.add_argument('output', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--budget-mib', type=int, required=True)
    parser.add_argument('--system-limit-mib', type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest() == manifest['workload_sha256']
    assert hashlib.sha256(Path(os.environ['LD_PRELOAD']).read_bytes()).hexdigest() == manifest['probe_sha256']
    config = workload.POLICY['index_build']
    assert (config['rows'], config['dimension'], config['build_workers']) == (400000, 768, 4)
    group = cgroup_path(os.getpid())
    assert int((group / 'memory.max').read_text()) == args.system_limit_mib * 1024**2
    assert int((group / 'memory.swap.max').read_text()) == 0
    identity = dict(base_commit=manifest['base_commit'], platform=platform.platform(),
                    cpuinfo=Path('/proc/cpuinfo').read_text(), cpu_max=(group / 'cpu.max').read_text(),
                    affinity=sorted(os.sched_getaffinity(0)), memory_max=(group / 'memory.max').read_text(),
                    budget_mib=args.budget_mib, system_limit_mib=args.system_limit_mib,
                    rows=400000, dimension=768, build_workers=4, seed=2026, batch_size=128,
                    max_degree=32, list_size=128, pq_chunk_num=0,
                    workload_sha256=hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest(),
                    probe_sha256=hashlib.sha256(Path(os.environ['LD_PRELOAD']).read_bytes()).hexdigest(),
                    cgroup_events_before=key_values(group / 'memory.events'))
    # Do not preload the observer into calibration/RSS-observer child processes.
    # The measured process has already loaded it; its wrappers remain active.
    os.environ.pop('LD_PRELOAD')
    # Independently fingerprint the exact fixed generator, outside measurement.
    digest = hashlib.sha256()
    rng = workload.np.random.default_rng(2026)
    for start in range(0, 400000, 128):
        vectors = rng.standard_normal((min(128, 400000 - start), 768), dtype=workload.np.float32)
        digest.update(vectors.tobytes())
    identity['corpus_float32_sha256'] = digest.hexdigest()
    del vectors, rng
    dump(output / 'identity.json', identity)
    os.environ['ZVEC_BUILD_RESOURCE_STATE'] = str(output / 'budget-state.json')
    original_init = zvec.init
    native = None
    def observed_init(*a, **kw):
        nonlocal native
        result = original_init(*a, **kw)
        assert native is None
        native = configure_probe(manifest, output, args.budget_mib)
        return result
    zvec.init = observed_init
    began = time.monotonic()
    try:
        with (output / 'build.jsonl').open('x') as evidence:
            with workload.ProcessRssMonitor(evidence, args.budget_mib * 1024):
                try:
                    workload.run('diskann', args.collection, evidence,
                                 400000, 768, args.budget_mib, 4)
                finally:
                    if native is not None:
                        native.zvec_build_probe_stop()
        identity['completed'] = True
    finally:
        zvec.init = original_init
        identity['elapsed_seconds'] = time.monotonic() - began
        identity['cgroup_events_after'] = key_values(group / 'memory.events')
        dump(output / 'identity.json', identity)
    assert identity['completed']
    for event in ['oom', 'oom_kill', 'oom_group_kill']:
        assert identity['cgroup_events_after'].get(event, 0) == identity['cgroup_events_before'].get(event, 0)

if __name__ == '__main__':
    main()

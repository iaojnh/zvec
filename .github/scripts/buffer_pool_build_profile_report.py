"""Summarize observed stages, retaining CPU time and waiting thread time separately."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def delta(a, b, *keys):
    for key in keys:
        a, b = a.get(key, {}), b.get(key, {})
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    assert b >= a, f'counter decreased: {keys}'
    return b - a

def summarize(folder):
    identity = json.loads((folder / 'identity.json').read_text())
    samples = rows(folder / 'build.samples.resources.jsonl')
    phases = rows(folder / 'build.jsonl')
    assert identity.get('completed'), 'workload incomplete'
    assert len(samples) >= 2 and all(b['monotonic_seconds'] > a['monotonic_seconds'] for a, b in zip(samples, samples[1:]))
    result = []
    names = ['start', 'insert_finished', 'flush_finished', 'build_started', 'build_finished',
             'closed_after_build', 'reopened_verified', 'complete']
    marks = {p['phase']: p for p in phases}
    for begin, end in zip(names, names[1:]):
        t0, t1 = (marks[n]['monotonic_seconds'] for n in [begin, end])
        if t1 - t0 < 2:
            continue
        a = min(samples, key=lambda r: abs(r['monotonic_seconds'] - t0))
        b = min(samples, key=lambda r: abs(r['monotonic_seconds'] - t1))
        assert abs(a['monotonic_seconds'] - t0) < 2 and abs(b['monotonic_seconds'] - t1) < 2
        wall = b['monotonic_seconds'] - a['monotonic_seconds']
        user = delta(a, b, 'process_user_ns') / 1e9
        system = delta(a, b, 'process_system_ns') / 1e9
        budget = {k: delta(a, b, 'budget', k) for k in [
            'cv_wait_calls', 'cv_wait_completed', 'cv_wait_ns', 'cv_wait_timeouts',
            'capacity_waits', 'capacity_wait_timeouts', 'high_watermark_hits',
            'bg_evict_rounds', 'bg_evicted_buffers', 'bg_no_progress_sleeps', 'slab_reclaimed_pages']}
        result.append(dict(begin=begin, end=end, phase_seconds=t1-t0,
            sampled_seconds=wall, boundary_error_seconds=[a['monotonic_seconds']-t0, b['monotonic_seconds']-t1],
            user_cpu_seconds=user, system_cpu_seconds=system, average_cpu_cores=(user+system)/wall,
            physical_read_bytes=delta(a, b, 'process_io', 'read_bytes'),
            physical_write_bytes=delta(a, b, 'process_io', 'write_bytes'),
            logical_read_bytes=delta(a, b, 'process_io', 'rchar'),
            io_some_stall_us=delta(a,b,'cgroup_io_psi','some_total_us'),
            io_full_stall_us=delta(a,b,'cgroup_io_psi','full_total_us'),
            memory_some_stall_us=delta(a,b,'cgroup_memory_psi','some_total_us'),
            memory_full_stall_us=delta(a,b,'cgroup_memory_psi','full_total_us'),
            cpu_throttled_us=delta(a,b,'cgroup_cpu','throttled_usec'),
            file_refaults=delta(a,b,'cgroup_memory','workingset_refault_file'),
            major_faults=delta(a,b,'major_faults'), budget=budget))
        result[-1]['cgroup_io_devices'] = {
            device: {key: delta(a, b, 'cgroup_io_devices', device, key)
                     for key in values}
            for device, values in b['cgroup_io_devices'].items()
        }
    native = [s['budget'] for s in samples if 'budget' in s]
    assert native
    if native[-1]['capacity_wait_timeouts'] > native[0]['capacity_wait_timeouts']:
        assert native[-1]['cv_wait_calls'] > 0, 'budget wait interceptor missed native waits'
    substages = {}
    for a, b in zip(samples, samples[1:]):
        if 'budget' not in a or 'budget' not in b:
            continue
        name = a.get('build_substage', 'unknown')
        if name not in substages:
            substages[name] = dict(sampled_seconds=0, cpu_seconds=0,
                physical_read_bytes=0, physical_write_bytes=0,
                io_some_stall_us=0, io_full_stall_us=0, memory_some_stall_us=0,
                budget_cv_wait_ns=0, budget_cv_wait_calls=0,
                capacity_wait_timeouts=0, high_watermark_hits=0,
                bg_evicted_buffers=0)
        value = substages[name]
        value['sampled_seconds'] += b['monotonic_seconds'] - a['monotonic_seconds']
        value['cpu_seconds'] += (delta(a,b,'process_user_ns') + delta(a,b,'process_system_ns')) / 1e9
        for label, keys in {
            'physical_read_bytes': ('process_io','read_bytes'),
            'physical_write_bytes': ('process_io','write_bytes'),
            'io_some_stall_us': ('cgroup_io_psi','some_total_us'),
            'io_full_stall_us': ('cgroup_io_psi','full_total_us'),
            'memory_some_stall_us': ('cgroup_memory_psi','some_total_us'),
            'budget_cv_wait_ns': ('budget','cv_wait_ns'),
            'budget_cv_wait_calls': ('budget','cv_wait_calls'),
            'capacity_wait_timeouts': ('budget','capacity_wait_timeouts'),
            'high_watermark_hits': ('budget','high_watermark_hits'),
            'bg_evicted_buffers': ('budget','bg_evicted_buffers'),
        }.items():
            difference = delta(a,b,*keys)
            if difference is None:
                value[label] = None
            elif value[label] is not None:
                value[label] += difference
    for value in substages.values():
        value['average_cpu_cores'] = value['cpu_seconds'] / value['sampled_seconds']
    return dict(identity=identity, stages=result, native_log_substages=substages, sampling_interval_seconds=1,
        maximum_sampling_gap_seconds=max(b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(samples,samples[1:])),
        caveats=['CV waiting is summed thread time, not disjoint wall time.',
                 'cgroup PSI/device IO include the supervisor and independent observer; process IO/CPU are workload-only.',
                 'Logical and cgroup memory limits are both recorded; changing both does not isolate a pure pool-capacity effect.',
                 'The bounded native sampler contributes a 128 KiB stack and a 4 KiB shared mapping.',
                 'Full construction stages come from unchanged harness markers; substage timings remain in build.log.'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    summary = summarize(args.folder)
    (args.folder / 'resource-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

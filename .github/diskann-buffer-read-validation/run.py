import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=pathlib.Path, required=True)
p.add_argument('--candidate', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--trials', type=int, default=12)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
protocol = pathlib.Path(__file__).resolve().parent
iswin = platform.system() == 'Windows'
suffix = '.exe' if iswin else ''
bench = 'diskann_buffer_read_bench_test'
reader = 'diskann_file_reader_windows_test' if iswin else 'diskann_file_reader_test'
builds = {}
rows = []

def invoke(args, log, **kwargs):
    with (out / log).open('w') as stream:
        subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True, **kwargs)

def environment(label):
    build = builds[label]
    env = {**os.environ}
    paths = [str(build / name) for name in ['bin', 'lib', 'lib64']]
    env['PATH'] = os.pathsep.join(paths + [env.get('PATH', '')])
    if not iswin:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(paths)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[name] = '1'
    return env

for label, source in [('baseline', a.baseline), ('candidate', a.candidate)]:
    source = source.resolve()
    injected = source / 'tests/core/interface' / (bench + '.cc')
    injected.write_bytes((protocol / 'diskann_read_bench.cc').read_bytes())
    test_cmake = source / 'tests/core/interface/CMakeLists.txt'
    text = test_cmake.read_text()
    text += '\nif(WIN32)\n  target_link_libraries(' + bench + ' psapi)\nendif()\n'
    test_cmake.write_text(text)
    build = out / ('build-' + label)
    builds[label] = build
    args = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_PYTHON_BINDINGS=OFF',
            '-DBUILD_C_BINDINGS=OFF', '-DBUILD_CPP_EXAMPLES=OFF', '-DBUILD_TOOLS=ON',
            '-DBUILD_ZVEC_SHARED=OFF', '-DBUILD_ZVEC_CORE_SHARED=OFF',
            '-DBUILD_ZVEC_AILEGO_SHARED=OFF', '-DENABLE_WERROR=ON']
    invoke(args, label + '-configure.log', timeout=900)
    invoke(['cmake', '--build', str(build), '--parallel', str(min(4, os.cpu_count() or 2)),
            '--target', reader, 'diskann_cache_test', 'buffer_read_storage_test', bench],
           label + '-build.log', timeout=1800)
    for name in [reader, 'diskann_cache_test', 'buffer_read_storage_test']:
        invoke([str(build / 'bin' / (name + suffix)),
                '--gtest_output=xml:' + str(out / (label + '-' + name + '.xml'))],
               label + '-' + name + '.log', cwd=out, env=environment(label), timeout=300)
    print('BUILT_AND_TESTED', label, flush=True)

def run(label, seed, workers, cold, trial, passes, prepare=False):
    env = environment(label)
    env.update(ZVEC_READ_BENCH_DIR=str(out / 'fixture'),
               ZVEC_READ_BENCH_QUERY_CACHE=str(out / 'query-data'),
               ZVEC_READ_BENCH_SEED=str(seed), ZVEC_READ_BENCH_BUDGET='128',
               ZVEC_READ_BENCH_WORKERS=str(workers), ZVEC_READ_BENCH_PASSES=str(passes),
               ZVEC_READ_BENCH_COLD=str(int(cold)), ZVEC_READ_BENCH_STEADY='1',
               ZVEC_READ_BENCH_PREPARE=str(int(prepare)))
    name = f'{label}-{seed}-w{workers}-cold{int(cold)}-{trial}'
    log = name + '.log'
    invoke([str(builds[label] / 'bin' / (bench + suffix)),
            '--gtest_output=xml:' + str(out / (name + '.xml'))],
           log, env=env, cwd=out, timeout=300)
    if prepare:
        return None
    lines = [line[len('READ_BENCH '):] for line in (out / log).read_text(errors='replace').splitlines()
             if line.startswith('READ_BENCH ')]
    assert len(lines) == 1, log
    row = json.loads(lines[0])
    assert row['page_bytes'] == 4096 and row['errors'] == 0 and row['recall'] >= .9, row
    assert row['budget_bytes'] == 128 << 20 and row['cpu_seconds'] > 0, row
    if not cold:
        assert row['steady_workers'] and row['cache_misses'] == 0 and row['bypass_bytes'] == 0, row
    row.update(label=label, trial=trial, log=log)
    rows.append(row)
    (out / 'measurements.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
    return row

for seed in [721, 2027]:
    run('baseline', seed, 1, False, 'prepare', 1, prepare=True)
    # Predeclared warm runs validate shared fixtures and DLL loading before timing.
    warm = [run(label, seed, 4, False, 'warm', 1) for label in ['baseline', 'candidate']]
    assert warm[0]['digest'] == warm[1]['digest']
    for workers, cold, trials, passes in [(4, False, a.trials, 64), (1, False, 5, 16), (4, True, 5, 4)]:
        for trial in range(trials):
            order = ['baseline', 'candidate']
            if trial % 2:
                order.reverse()
            pair = {label: run(label, seed, workers, cold, trial, passes) for label in order}
            assert pair['baseline']['digest'] == pair['candidate']['digest'], pair
            assert pair['baseline']['recall'] == pair['candidate']['recall'], pair
            if workers == 4 and not cold and trial in [0, 5]:
                controls = [run('baseline', seed, workers, cold, f'AA-{trial}-{side}', passes)
                            for side in ['first', 'second']]
                assert controls[0]['digest'] == controls[1]['digest']

summary = []
rng = random.Random(20261110)
for seed in [721, 2027]:
    for workers, cold in [(4, False), (1, False), (4, True)]:
        group = [x for x in rows if x['seed'] == seed and x['workers'] == workers
                 and bool(x['cold_managed_cache']) == cold and isinstance(x['trial'], int)]
        by = {label: sorted((x for x in group if x['label'] == label), key=lambda x: x['trial'])
              for label in ['baseline', 'candidate']}
        b, c = by['baseline'], by['candidate']
        item = {'seed': seed, 'workers': workers, 'cold': cold, 'pairs': len(b)}
        for metric in ['qps', 'p50_us', 'p99_us', 'peak_rss', 'pool_used', 'cpu_seconds']:
            item[metric + '_baseline'] = statistics.median(x[metric] for x in b)
            item[metric + '_candidate'] = statistics.median(x[metric] for x in c)
        ratios = [y['qps'] / x['qps'] for x, y in zip(b, c)]
        boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(6000))
        item['paired_qps_ratio_95_bootstrap'] = [boot[150], boot[5850]]
        item['qps_ratio'] = item['qps_candidate'] / item['qps_baseline']
        item['cpu_ratio'] = item['cpu_seconds_candidate'] / item['cpu_seconds_baseline']
        summary.append(item)
primary = [x for x in summary if x['workers'] == 4 and not x['cold']]
qualified = all(x['paired_qps_ratio_95_bootstrap'][0] > 1 and x['cpu_ratio'] <= 1 for x in primary)
provenance = {'platform': platform.platform(), 'machine': platform.machine(), 'native_page_bytes': 4096,
              'sources': {label: {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                                  'src_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], cwd=source, text=True).strip()}
                          for label, source in [('baseline', a.baseline), ('candidate', a.candidate)]},
              'binaries': {label: hashlib.sha256((build / 'bin' / (bench + suffix)).read_bytes()).hexdigest()
                           for label, build in builds.items()},
              'entry': 'C++ DiskANN Buffer, 20000 x 256 FP32, R32/PQ16, list512/top10, production 128 MiB pool',
              'limitations': ['Synthetic clustered data, two seeds.', 'Cold is managed-cache cold; OS cache is retained.'],
              'merged': False}
(out / 'report.json').write_text(json.dumps({'complete': True, 'qualified': qualified,
                                            'provenance': provenance, 'summary': summary}, indent=2))
print('COMPLETED', json.dumps({'qualified': qualified, 'summary': summary}), flush=True)
if not qualified:
    raise SystemExit('Performance benefit is not established; do not merge.')

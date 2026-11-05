"""Verify Apple production-reader cache eviction and actual query storage I/O."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
import benchmark_diskann_platform as benchmark
import apple_io_monitor


def run_test(args, binary, name, probe=False):
    token = uuid.uuid4().hex
    env = os.environ.copy()
    if args.transport == 'ios':
        identifier = 'com.zvec.' + binary.stem
        benchmark.run(['xcrun', 'simctl', 'install', args.device, str(binary)])
        container = benchmark.run(['xcrun', 'simctl', 'get_app_container', args.device,
                                   identifier, 'data'], capture_output=True).stdout.strip()
        root = Path(container) / 'Documents' / ('apple-io-' + token)
        command = ['xcrun', 'simctl', 'launch', '--console', '--terminate-running-process',
                   args.device, identifier]
    else:
        root = args.output.resolve() / ('native-' + token)
        command = [str(binary.resolve())]
    root.mkdir(parents=True, exist_ok=True)
    test_result, result = root / 'gtest.json', root / 'result.txt'
    command += ['--gtest_output=json:' + str(test_result)]
    variables = {'ZVEC_APPLE_IO_RESULT_PATH': str(result)}
    if probe:
        variables['ZVEC_APPLE_IO_PROBE'] = '1'
    control = root / 'control'
    if probe and args.transport == 'ios':
        variables['ZVEC_APPLE_IO_CONTROL'] = str(control)
    env.update({('SIMCTL_CHILD_' if args.transport == 'ios' else '') + k: v
                for k, v in variables.items()})
    path = args.output / (name + '.log')
    observations = {}
    with path.open('w') as log:
        if probe and args.transport == 'ios':
            completed, observations = apple_io_monitor.run(command, env, log, control, test_result)
        else:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, timeout=600)
    deadline = time.monotonic() + 60
    while args.transport == 'ios' and not test_result.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    text = path.read_text(errors='replace')
    race = args.transport == 'ios' and completed.returncode == 3 and 'FBSOpenApplicationServiceErrorDomain' in text
    if (completed.returncode and not race) or not test_result.exists():
        print(text[-16000:])
        raise RuntimeError('Apple test process failed: ' + name)
    tests = json.loads(test_result.read_text())
    if not tests['tests'] or tests['failures'] or tests.get('errors', 0):
        print(text[-16000:])
        raise RuntimeError('Apple regression assertions failed: ' + name)
    (args.output / (name + '-gtest.json')).write_text(json.dumps(tests, indent=2) + '\n')
    if not probe:
        return None
    matches = re.findall(r'ZVEC_APPLE_IO_RESULT (\{[^\r\n]+\})', result.read_text())
    if len(matches) != 1:
        raise ValueError('Missing durable Apple I/O probe result')
    report = json.loads(matches[0])
    for case in report['cases']:
        if args.transport == 'ios':
            measured = observations[case['state']]
            case['read_bytes'] = measured['read_bytes']
            case['counter_source'] = measured['source']
        else:
            case['read_bytes'] = case['io_after'] - case['io_before']
            case['counter_source'] = 'native proc_pid_rusage OS disk-I/O bytes'
    return report


def evaluate(report):
    cases = {c['state']: c for c in report['cases']}
    if len(cases) != 6 or not report['payload_validated']:
        raise ValueError('Incomplete Apple cache probe')
    cold = cases['production-cold']
    buffered = cases['buffered-cold']
    controls = (cases['buffered-warm']['resident_before'] >= report['fixture_bytes'] * .9
                and buffered['resident_before'] <= report['fixture_bytes'] * .01
                and buffered['resident_after'] >= buffered['requested_bytes'] * .9
                and buffered['read_bytes'] >= buffered['requested_bytes'] * .9)
    verified = (controls and cold['resident_before'] <= report['fixture_bytes'] * .01
                and cold['resident_after'] <= report['fixture_bytes'] * .01)
    for c in report['cases']:
        if c['operations'] != 4096:
            raise ValueError('Missing successful reads')
        if c['state'].startswith('production'):
            verified = (verified and c['invalidation_requested'] and not c['fallback_errno']
                        and c['read_bytes'] >= c['requested_bytes'] * .9)
            if c['state'].endswith('warm'):
                verified = verified and c['resident_before'] >= report['fixture_bytes'] * .9
    report['verification'] = {'controls_valid': controls, 'cache_bypass_verified': bool(verified),
                              'method': 'bounded synchronous pre/post cache invalidation; F_NOCACHE where independent',
                              'scope': 'measured file-cache behavior, not an atomic OS O_DIRECT guarantee or phone flash'}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--transport', choices=['native', 'ios'], default='native')
    parser.add_argument('--device', default='booted')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_test(args, args.regression, 'regression')
    report = evaluate(run_test(args, args.probe, 'apple-io', probe=True))
    report['platform'] = args.label
    report['host'] = benchmark.metadata()
    (args.output / 'cache-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Apple DiskANN cache verification', '',
             '64 MiB nonzero fixture; 4096 aligned 4 KiB reads per case; every byte checked. No system-wide purge.', '',
             '| Case | Storage read MiB | Cached MiB before | Cached MiB after | Invalidation active | Fallback errno |',
             '| --- | --- | --- | --- | --- | --- |']
    for c in report['cases']:
        lines.append(f'| {c["state"]} | {c["read_bytes"]/1048576:.3f} | {c["resident_before"]/1048576:.3f} | '
                     f'{c["resident_after"]/1048576:.3f} | {c["invalidation_requested"]} | {c["fallback_errno"]} |')
    lines += ['', 'Verified measured cache bypass: **' + str(report['verification']['cache_bypass_verified']) + '**', '',
              'iOS observations use the simulator app process disk counter on its macOS host. No physical-phone claim. Other buffered readers can race range invalidation; this is not an OS-level atomic direct-I/O guarantee.', '']
    (args.output / 'verification.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)
    if not report['verification']['cache_bypass_verified']:
        raise RuntimeError('Apple production-reader cache bypass was not verified')
    command = [sys.executable, 'scripts/benchmark_diskann_platform.py', '--binary', str(args.benchmark),
               '--data', str(args.data), '--output', str(args.output / 'queries'), '--label', args.label,
               '--transport', args.transport, '--device', args.device, '--cache-comparison']
    if args.transport == 'ios':
        command += ['--monitor-ios-io']
    benchmark.run(command)
    comparison = json.loads((args.output / 'queries/cache-comparison.json').read_text())
    for case in comparison['cases']:
        if case['algorithm'] == 'diskann' and case['state'] == 'steady':
            q = case['query']
            io = q['io'].get('host_timed_read_bytes', q['io']['timed_read_bytes'])
            if io is None or io < q['queries'] * q['repetitions'] * 4096:
                raise RuntimeError('Warmed DiskANN search did not demonstrate continuing storage I/O')


if __name__ == '__main__':
    main()

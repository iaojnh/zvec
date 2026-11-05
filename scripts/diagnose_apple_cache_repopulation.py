"""Controlled Darwin cache attribution, independent of the benchmark's timings.

Reproduce the production syscalls using ctypes; compare linked/unlinked files,
transient/held observation mappings, and buffered/production follow-up reads.
The separate C++ production-reader CI remains the acceptance test.
"""
from __future__ import annotations
import argparse
import ctypes as c
import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time


class Probe:
    size = 64 * 1048576
    reads = 4096
    block = 4096

    def __init__(self, directory, anonymous, held):
        self.lib = c.CDLL(None, use_errno=True)
        self.proc = c.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.lib.mmap.argtypes = [c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_int, c.c_longlong]
        self.lib.mmap.restype = c.c_void_p
        self.lib.munmap.argtypes = [c.c_void_p, c.c_size_t]
        self.lib.msync.argtypes = [c.c_void_p, c.c_size_t, c.c_int]
        self.lib.mincore.argtypes = [c.c_void_p, c.c_size_t, c.c_void_p]
        self.lib.pread.argtypes = [c.c_int, c.c_void_p, c.c_size_t, c.c_longlong]
        self.lib.pread.restype = c.c_longlong
        self.proc.proc_pid_rusage.argtypes = [c.c_int, c.c_int, c.c_void_p]
        self.page = os.sysconf('SC_PAGESIZE')
        self.buffer = c.c_void_p()
        assert self.lib.posix_memalign(c.byref(self.buffer), self.page, self.page) == 0
        self.source, self.path = tempfile.mkstemp(prefix='ZvecCacheFixture.', dir=directory)
        self.raw = os.open(self.path, os.O_RDONLY)
        self.direct = os.open(self.path, os.O_RDONLY)
        if anonymous:
            # No content exists yet; another process cannot subsequently open
            # this inode by its pathname. Keep all independent handles alive.
            os.unlink(self.path)
        for fd in (self.source, self.raw, self.direct):
            fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD
        fcntl.fcntl(self.direct, 48, 1)  # F_NOCACHE, per description
        for block in range(self.size // self.block):
            payload = struct.pack('Q', 0x47a91bd39e510000 + block) * (self.block // 8)
            assert os.write(self.source, payload) == self.block
        os.fsync(self.source)
        self.mapping = self.map(0, self.size, 0) if held else None
        self.stats = {}
        self.result = {'anonymous': anonymous, 'held_mapping': held, 'pid': os.getpid(),
                       'fixture': self.path, 'page_size': self.page, 'cases': []}

    def map(self, offset, length, protection):
        ptr = self.lib.mmap(None, length, protection, mmap.MAP_SHARED, self.direct, offset)
        if ptr == c.c_void_p(-1).value:
            raise OSError(c.get_errno(), 'mmap')
        return ptr

    def bits(self, ptr, length):
        vector = (c.c_ubyte * ((length + self.page - 1) // self.page))()
        if self.lib.mincore(ptr, length, vector):
            raise OSError(c.get_errno(), 'mincore')
        return bytes(vector)

    def snapshot(self):
        ptr = self.mapping or self.map(0, self.size, 0)
        try:
            return self.bits(ptr, self.size)
        finally:
            if self.mapping is None:
                assert self.lib.munmap(ptr, self.size) == 0

    def disk_bytes(self):
        data = c.create_string_buffer(1024)
        assert self.proc.proc_pid_rusage(os.getpid(), 4, data) == 0
        return c.c_uint64.from_buffer(data, 144).value

    def evict(self, offset, length, point):
        begin = offset // self.page * self.page
        end = (offset + length + self.page - 1) // self.page * self.page
        for current in range(begin, end, 1048576):
            count = min(1048576, end - current)
            ptr = self.map(current, count, mmap.PROT_READ)
            try:
                present = any(v & 1 for v in self.bits(ptr, count))
                if present:
                    self.stats[point + '_initially_present'] = self.stats.get(point + '_initially_present', 0) + 1
                for attempt in range(8):
                    if self.lib.msync(ptr, count, 0x10 | 0x2):
                        raise OSError(c.get_errno(), 'msync')
                    if not any(v & 1 for v in self.bits(ptr, count)):
                        break
                    self.stats['retry'] = self.stats.get('retry', 0) + 1
                    time.sleep(.00005)
                else:
                    raise RuntimeError('EBUSY: resident pages after eight invalidations')
            finally:
                assert self.lib.munmap(ptr, count) == 0

    def read(self, fd, block):
        assert self.lib.pread(fd, self.buffer, self.block, block * self.block) == self.block
        expected = struct.pack('Q', 0x47a91bd39e510000 + block) * (self.block // 8)
        assert c.string_at(self.buffer, self.block) == expected

    def record(self, name, operation):
        # Named marker opens identify phase boundaries in fs_usage.
        marker = Path(self.path).parent / ('ZvecCachePhase.' + name)
        marker.touch()
        self.stats = {}
        before = self.snapshot()
        start = self.disk_bytes()
        operation()
        read_bytes = self.disk_bytes() - start
        samples = []
        for pause in (0, .01, .1, .5):
            time.sleep(pause)
            bits = self.snapshot()
            samples.append({'delay': pause, 'resident_bytes': sum(bool(v & 1) for v in bits) * self.page})
        self.result['cases'].append({'name': name, 'read_bytes': read_bytes, 'stats': self.stats,
                                     'resident_before': sum(bool(v & 1) for v in before) * self.page,
                                     'samples': samples})
        marker.unlink()

    def loop(self, production=False, no_read=False):
        for i in range(self.reads):
            block = (i * 1009) % (self.size // self.block)
            offset = block * self.block
            if production:
                self.evict(offset, self.block, 'pre')
            if not no_read:
                self.read(self.direct, block)
            if production:
                self.evict(offset, self.block, 'post')

    def followup(self, production):
        bits = self.snapshot()
        blocks = [i for i in range(self.size // self.block) if bits[i * self.block // self.page] & 1][:128]
        self.stats['selected_resident_blocks'] = len(blocks)
        for block in blocks:
            if production:
                self.evict(block * self.block, self.block, 'pre')
            self.read(self.direct if production else self.raw, block)
            if production:
                self.evict(block * self.block, self.block, 'post')

    def run(self):
        # Let initial writeback settle; only this probe's ranges are evicted.
        time.sleep(.5)
        self.evict(0, self.size, 'prepare')
        self.record('idle-after-eviction', lambda: None)
        self.record('invalidation-only', lambda: self.loop(production=True, no_read=True))
        self.evict(0, self.size, 'prepare')
        self.record('hint-only-cold', self.loop)
        self.evict(0, self.size, 'prepare')
        self.record('production-cold', lambda: self.loop(production=True))
        self.record('production-repeat', lambda: self.loop(production=True))
        self.record('resident-buffered-followup', lambda: self.followup(False))
        self.record('resident-production-followup', lambda: self.followup(True))
        # Deliberately contaminated positive control: a later production read
        # must miss cache even when an ordinary reader just populated it.
        for block in range(128):
            self.read(self.raw, block)
        self.record('seeded-buffered-followup', lambda: [self.read(self.raw, b) for b in range(128)])
        self.record('seeded-production-followup', lambda: [
            (self.evict(b * self.block, self.block, 'pre'), self.read(self.direct, b),
             self.evict(b * self.block, self.block, 'post')) for b in range(128)])
        return self.result

    def close(self):
        if self.mapping is not None:
            self.lib.munmap(self.mapping, self.size)
        for fd in (self.source, self.raw, self.direct):
            os.close(fd)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.lib.free(self.buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = None
    retained = []
    if args.trace:
        if os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted':
            raise RuntimeError('Privileged tracing is restricted to ephemeral GitHub-hosted CI')
        trace = subprocess.Popen(['sudo', '-n', '/usr/bin/fs_usage', '-w', '-f', 'filesys',
                                  '-f', 'diskio', '-f', 'cachehit', '-t', '120'],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 errors='replace')
        def collect():
            with (args.output / 'fs-usage-fixture.log').open('w') as out:
                for line in trace.stdout:
                    if len(retained) < 30:
                        retained.append(line)
                    if 'ZvecCache' in line or 'Python.' in line or 'python' in line.lower():
                        out.write(line)
        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        time.sleep(1)
    report = {'pid': os.getpid(), 'variants': [], 'trace_requested': args.trace}
    try:
        for anonymous in (False, True):
            for held in (False, True):
                with tempfile.TemporaryDirectory(prefix='ZvecCache.', dir=os.getcwd()) as directory:
                    probe = Probe(directory, anonymous, held)
                    try:
                        try:
                            result = probe.run()
                        except Exception as error:
                            result = probe.result
                            result['error'] = repr(error)
                            result['last_stats'] = probe.stats
                        report['variants'].append(result)
                        print(json.dumps(result), flush=True)
                        (args.output / 'attribution.json').write_text(json.dumps(report, indent=2) + '\n')
                    finally:
                        probe.close()
    finally:
        if trace is not None:
            if trace.poll() is None:
                subprocess.run(['sudo', '-n', '/bin/kill', '-INT', str(trace.pid)], check=False)
            trace.wait(timeout=130)
            reader.join(timeout=5)
            report['trace_exit'] = trace.returncode
            report['trace_header'] = retained
        (args.output / 'attribution.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

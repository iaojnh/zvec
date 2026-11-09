"""Low-rate Linux resource observation and exact diagnostic ELF validation."""
from __future__ import annotations
import hashlib
import json
import mmap
import os
import re
from pathlib import Path
import struct
import time

MAGIC = 0x5A56454342505231
WAIT_NAMES = ('magic', 'sequence', 'sample_ns', 'cv_wait_calls', 'cv_wait_completed',
              'cv_wait_ns', 'cv_wait_timeouts', 'cv_wait_active', 'cv_max_wait_ns')
STAGES = [('Begin DiskAnnBuilder::init', 'initialization'),
          ('Begin DiskAnnBuilder::train', 'pq_training'),
          ('Start DiskAnnBuilder::build', 'vector_staging'),
          ('Finished saving vector', 'pq_encoding'),
          ('Generate Quantized Data Done', 'graph_code_staging'),
          ('Start to calculate entrypoint', 'entrypoint'),
          ('DiskANN graph build pass 1/2', 'graph_pass_1'),
          ('DiskANN graph build pass 2/2', 'graph_pass_2'),
          ('Start final cleanup', 'prune_and_repair'),
          ('DiskANN connectivity repair added', 'compaction'),
          ('End DiskAnnBuilder::build', 'builder_finished'),
          ('Begin DiskAnnBuilder::dump', 'index_dump')]

def log_stage(line):
    for text, name in STAGES:
        if text in line:
            return name
    return None

def elf_identity(path):
    """Hash all loadable bytes, ignoring only strip's section-table metadata."""
    with Path(path).open('rb') as file:
        data = file.read(64)
        assert data[:6] == b'\x7fELF\x02\x01', 'requires little-endian ELF64'
        header = struct.unpack('<16sHHIQQQIHHHHHH', data)
        file.seek(header[5])
        program_headers = file.read(header[10] * header[9])
    assert header[1] == 3, 'requires an ELF shared object'
    digest = hashlib.sha256()
    segments = []
    for i in range(header[10]):
        ph = struct.unpack_from('<IIQQQQQQ', program_headers, i * header[9])
        kind, flags, offset, address, _, size, memory, alignment = ph
        if kind != 1:
            continue
        shape = (flags, address, size, memory, alignment)
        digest.update(struct.pack('<5Q', *shape))
        with Path(path).open('rb') as file:
            file.seek(offset)
            remaining = size
            first = True
            while remaining:
                contents = bytearray(file.read(min(65536, remaining)))
                assert contents, 'truncated loadable ELF segment'
                if first and offset == 0:
                    contents[40:48] = b'\0' * 8
                    contents[60:64] = b'\0' * 4
                digest.update(contents)
                remaining -= len(contents)
                first = False
        segments.append(dict(flags=flags, vaddr=address, filesz=size, memsz=memory))
    assert segments and segments[0]['vaddr'] == 0
    return dict(machine=header[2], loadable_sha256=digest.hexdigest(), segments=segments)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()

def key_values(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        parts = line.replace(':', '').split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result

def psi(path):
    result = {}
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return result
    for line in lines:
        parts = line.split()
        values = dict(item.split('=', 1) for item in parts[1:])
        result[parts[0] + '_total_us'] = int(values['total'])
    return result

def cgroup_path(pid):
    lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    relative = next(line.split(':', 2)[2] for line in lines if line.startswith('0::'))
    return Path('/sys/fs/cgroup') / relative.lstrip('/')

def read_budget(path, names):
    with Path(path).open('rb') as file, mmap.mmap(file.fileno(), 4096, access=mmap.ACCESS_READ) as mapping:
        for _ in range(30):
            first = struct.unpack_from('<Q', mapping, 8)[0]
            values = struct.unpack_from('<25Q', mapping)
            last = struct.unpack_from('<Q', mapping, 8)[0]
            if first == last and first % 2 == 0:
                assert values[0] == MAGIC
                result = dict(zip(WAIT_NAMES, values[:9]))
                result.update(zip(names, values[9:9 + len(names)]))
                return result
        raise RuntimeError('budget observer did not publish a stable snapshot')

class ResourceCollector:
    def __init__(self, pid, state_path, output_path):
        self.pid = pid
        self.state_path = Path(state_path)
        self.output = Path(output_path).open('x')
        self.group = cgroup_path(pid)
        self.next_sample = 0.0
        self.hz = os.sysconf('SC_CLK_TCK')
        self.started = time.monotonic()
        self.phase_offset = 0
        self.phase = 'startup'
        self.state = None
        self.log_path = os.environ.get('ZVEC_BUILD_PROFILE_LOG')
        self.log_offset = 0
        self.substage = 'startup'
        self.progress = None

    def sample(self, now, force=False):
        if not force and now < self.next_sample:
            return
        self.next_sample = now + 1.0
        if self.log_path and Path(self.log_path).exists():
            with Path(self.log_path).open('rb') as file:
                file.seek(self.log_offset)
                content = file.read(1024**2)
            complete = content.rfind(b'\n') + 1
            self.log_offset += complete
            for line in content[:complete].decode('utf-8', errors='replace').splitlines():
                stage = log_stage(line)
                if stage:
                    self.substage = stage
                    self.progress = None
                match = re.search(r'Built cnt (\d+), finished percent ([\d.]+)%', line)
                if match:
                    self.progress = dict(nodes=int(match[1]), percent=float(match[2]), line=line)
        if self.state is None and self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        if self.state:
            with Path(self.state['phase_file']).open() as file:
                file.seek(self.phase_offset)
                while True:
                    line = file.readline()
                    if not line or not line.endswith('\n'):
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    self.phase = row.get('phase', self.phase)
                    self.phase_offset = file.tell()
        proc = Path(f'/proc/{self.pid}')
        raw = (proc / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        status = key_values(proc / 'status')
        row = dict(monotonic_seconds=now, phase=self.phase, pid=self.pid,
                   build_substage=self.substage, graph_progress=self.progress,
                   log_backlog_bytes=Path(self.log_path).stat().st_size-self.log_offset if self.log_path else None,
                   process_user_ns=int(fields[11]) * 10**9 // self.hz,
                   process_system_ns=int(fields[12]) * 10**9 // self.hz,
                   minor_faults=int(fields[7]), major_faults=int(fields[9]),
                   rss_kib=status.get('VmRSS'), threads=status.get('Threads'),
                   process_io=key_values(proc / 'io'),
                   cgroup_cpu=key_values(self.group / 'cpu.stat'),
                   cgroup_memory=key_values(self.group / 'memory.stat'),
                   cgroup_events=key_values(self.group / 'memory.events'),
                   cgroup_io_psi=psi(self.group / 'io.pressure'),
                   cgroup_memory_psi=psi(self.group / 'memory.pressure'),
                   cgroup_cpu_psi=psi(self.group / 'cpu.pressure'),
                   observer_io=key_values('/proc/self/io'),
                   observer_rss_kib=key_values('/proc/self/status').get('VmRSS'))
        devices = {}
        for line in (self.group / 'io.stat').read_text().splitlines():
            parts = line.split()
            devices[parts[0]] = {k: int(v) for k, v in (item.split('=') for item in parts[1:])}
        row['cgroup_io_devices'] = devices
        thread_states = []
        for task in sorted((proc / 'task').iterdir(), key=lambda p: int(p.name)):
            try:
                raw_task = (task / 'stat').read_text()
                task_fields = raw_task[raw_task.rfind(')') + 2:].split()
                thread_states.append(dict(tid=int(task.name), state=task_fields[0],
                    cpu_ticks=int(task_fields[11]) + int(task_fields[12]),
                    schedstat=(task / 'schedstat').read_text().strip(),
                    wchan=(task / 'wchan').read_text().strip()))
            except (FileNotFoundError, ProcessLookupError):
                continue
        assert len(thread_states) <= 128, 'diagnostic thread bound exceeded'
        row['thread_states'] = thread_states
        if self.state:
            row['budget'] = read_budget(self.state['shared_file'], self.state['field_names'])
            row['budget']['sample_age_seconds'] = now - row['budget']['sample_ns'] / 1e9
            assert row['budget']['sample_age_seconds'] < 5, 'native observer stalled'
            assert 0 < row['budget']['pool_size'] <= self.state['budget_mib'] * 1024**2
        print(json.dumps(row, separators=(',', ':')), file=self.output, flush=True)
        if now - self.started >= 60:
            self.started = now
            print('RESOURCE ' + json.dumps(row, separators=(',', ':')), flush=True)

    def close(self):
        self.output.close()

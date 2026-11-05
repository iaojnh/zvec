"""Observe a simulator app's host process, using public macOS libproc only."""
from __future__ import annotations
import ctypes
import json
import subprocess
import time


def disk_read_bytes(pid):
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    data = ctypes.create_string_buffer(1024)
    if library.proc_pid_rusage(pid, 4, data) != 0:
        raise OSError(ctypes.get_errno(), "Cannot observe simulator process disk I/O")
    # Public rusage_info_v4: UUID[16], sixteen uint64 fields, diskio_bytesread.
    return ctypes.c_uint64.from_buffer(data, 16 + 16 * 8).value


def run(command, env, log, control, completion, timeout=600):
    control.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    seen, starts, reports = set(), {}, {}
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            status = control / 'status.json'
            if status.exists():
                value = json.loads(status.read_text())
                key = (value['case'], value['point'])
                if key not in seen:
                    current = disk_read_bytes(value['pid'])
                    if value['point'] == 'before':
                        starts[value['case']] = current
                    else:
                        previous = starts[value['case']]
                        if current < previous:
                            raise ValueError("Simulator disk counters decreased")
                        reports[value['case']] = {'read_bytes': current - previous,
                                                  'pid': value['pid'],
                                                  'source': 'simulator-host proc_pid_rusage; not phone flash'}
                    seen.add(key)
                    (control / ('.'.join(key) + '.go')).write_text('go\n')
            code = process.poll()
            if code is not None and (completion.exists() or code not in (0, 3)):
                return subprocess.CompletedProcess(command, code), reports
            time.sleep(.02)
        raise TimeoutError("Simulator I/O verification timed out")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

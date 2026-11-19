"""Observe query-window storage activity outside the Linux IOPS sweep."""

import ctypes
from functools import cache
from pathlib import Path
import subprocess
import sys
import time

import psutil


@cache
def apple_proc_rusage():
    read = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
    read.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    read.restype = ctypes.c_int
    return read


def process_disk_bytes(pid):
    data = ctypes.create_string_buffer(1024)
    if apple_proc_rusage()(pid, 4, data) != 0:
        return None
    # rusage_info_v4 has a 16-byte UUID and 16 uint64 fields before diskio_bytesread.
    return ctypes.c_uint64.from_buffer(data, 16 + 16 * 8).value


def desktop_snapshot(pid):
    disks = psutil.disk_io_counters(perdisk=False, nowrap=False)
    logical_reads = logical_bytes = storage_bytes = None
    if sys.platform == "win32":
        try:
            counts = psutil.Process(pid).io_counters()
            logical_reads, logical_bytes = counts.read_count, counts.read_bytes
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    else:
        storage_bytes = process_disk_bytes(pid)
    return dict(
        device_read_ios=disks.read_count if disks is not None else None,
        device_read_bytes=disks.read_bytes if disks is not None else None,
        process_logical_read_ios=logical_reads,
        process_logical_read_bytes=logical_bytes,
        process_storage_read_bytes=storage_bytes,
    )


def difference(before, after):
    return {
        key: after[key] - value if value is not None and after[key] is not None and after[key] >= value
        else None
        for key, value in before.items()
    }


def desktop_observation(before, after, simulator=False):
    return dict(
        scope="macOS simulator host, all block devices" if simulator else "runner, all block devices",
        cache_policy=("FILE_FLAG_NO_BUFFERING for the DiskANN graph" if sys.platform == "win32"
                      else "F_NOCACHE requested by the DiskANN reader; advisory"),
        **difference(before, after),
    )


def run_ios_observed(command, env, log, control, result):
    """Read the simulator's host counters while its query worker waits at each boundary."""
    control.mkdir()
    env = {**env, "SIMCTL_CHILD_ZVEC_BENCH_IO_CONTROL": str(control)}
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    snapshots = {}
    sample_times = {}
    deadline = time.monotonic() + 3600
    exited_at = None
    try:
        while time.monotonic() < deadline:
            for point in ("before", "after"):
                marker = control / point
                if point not in snapshots and marker.is_file():
                    pid = int(marker.read_text())
                    snapshots[point] = desktop_snapshot(pid)
                    sample_times[point] = time.perf_counter_ns()
                    (control / f"{point}.go").touch()
            code = process.poll()
            if code is not None:
                if result.is_file() or code not in (0, 3):
                    break
                if exited_at is None:
                    exited_at = time.monotonic()
                if time.monotonic() - exited_at >= 15:
                    break
            time.sleep(0.02)
        else:
            raise TimeoutError("Simulator query I/O observation timed out")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    observation = None
    if snapshots.keys() >= {"before", "after"}:
        observation = desktop_observation(snapshots["before"], snapshots["after"], simulator=True)
        observation["counter_window_ns"] = sample_times["after"] - sample_times["before"]
    return process.returncode, observation

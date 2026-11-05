"""Read-only Windows 4K storage probes, with the same flags as DiskANN."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import time
from ctypes import wintypes
from pathlib import Path


class Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE)]


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = ctypes.c_void_p
    ulong_pointer = ctypes.c_size_t
    overlap_pointer = ctypes.POINTER(Overlapped)
    signatures = {
        "CreateFileW": ([wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, pointer,
                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE], wintypes.HANDLE),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "VirtualAlloc": ([pointer, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD], pointer),
        "VirtualFree": ([pointer, ctypes.c_size_t, wintypes.DWORD], wintypes.BOOL),
        "SetFilePointerEx": ([wintypes.HANDLE, ctypes.c_longlong, pointer, wintypes.DWORD], wintypes.BOOL),
        "ReadFile": ([wintypes.HANDLE, pointer, wintypes.DWORD,
                      ctypes.POINTER(wintypes.DWORD), overlap_pointer], wintypes.BOOL),
        "CreateIoCompletionPort": ([wintypes.HANDLE, wintypes.HANDLE, ulong_pointer,
                                    wintypes.DWORD], wintypes.HANDLE),
        "GetQueuedCompletionStatus": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                       ctypes.POINTER(ulong_pointer), ctypes.POINTER(overlap_pointer),
                                       wintypes.DWORD], wintypes.BOOL),
    }
    for name, (parameters, result) in signatures.items():
        getattr(k, name).argtypes = parameters
        getattr(k, name).restype = result
    invalid = ctypes.c_void_p(-1).value
    filename = str(args.file.resolve())
    blocks = args.file.stat().st_size // 4096
    if blocks < 16:
        raise ValueError("Probe file too small")
    # Warm the system file cache for the buffered control. Unbuffered cases
    # explicitly bypass that cache and use page-aligned VirtualAlloc memory.
    with args.file.open("rb") as source:
        while source.read(1024 * 1024):
            pass
    buffer = k.VirtualAlloc(None, 8 * 4096, 0x3000, 0x04)
    if not buffer:
        raise ctypes.WinError(ctypes.get_last_error())
    results = []

    def open_file(direct, async_io):
        flags = 0x80 | (0x20000000 if direct else 0) | (0x40000000 if async_io else 0)
        handle = k.CreateFileW(filename, 0x80000000, 7, None, 3, flags, None)
        if handle == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    try:
        for direct, async_io, depth in [(False, False, 1), (True, False, 1),
                                        (True, True, 1), (True, True, 8)]:
            handle = open_file(direct, async_io)
            port = k.CreateIoCompletionPort(handle, None, 0, 1) if async_io else None
            if async_io and not port:
                k.CloseHandle(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            rng = random.Random(20261105)
            latencies = []
            checksum = 0
            try:
                for _ in range(1024):
                    offsets = [rng.randrange(blocks) * 4096 for _ in range(depth)]
                    operations = (Overlapped * depth)()
                    byte_count = wintypes.DWORD()
                    start = time.perf_counter_ns()
                    for i, offset in enumerate(offsets):
                        if async_io:
                            operations[i].Offset = offset & 0xFFFFFFFF
                            operations[i].OffsetHigh = offset >> 32
                            success = k.ReadFile(handle, buffer + i * 4096, 4096, None,
                                                 ctypes.byref(operations[i]))
                            if not success and ctypes.get_last_error() != 997:
                                raise ctypes.WinError(ctypes.get_last_error())
                        else:
                            if not k.SetFilePointerEx(handle, offset, None, 0):
                                raise ctypes.WinError(ctypes.get_last_error())
                            if not k.ReadFile(handle, buffer, 4096, ctypes.byref(byte_count), None):
                                raise ctypes.WinError(ctypes.get_last_error())
                            if byte_count.value != 4096:
                                raise RuntimeError("Short synchronous read")
                    if async_io:
                        pending = {ctypes.addressof(op) for op in operations}
                        for _ in range(depth):
                            key = ulong_pointer()
                            completed = overlap_pointer()
                            if not k.GetQueuedCompletionStatus(port, ctypes.byref(byte_count),
                                                               ctypes.byref(key), ctypes.byref(completed),
                                                               0xFFFFFFFF):
                                raise ctypes.WinError(ctypes.get_last_error())
                            address = ctypes.cast(completed, pointer).value
                            if address not in pending or byte_count.value != 4096:
                                raise RuntimeError("Invalid IOCP completion")
                            pending.remove(address)
                    latencies.append((time.perf_counter_ns() - start) / 1e6)
                    checksum += ctypes.c_ubyte.from_address(buffer).value
            finally:
                k.CloseHandle(handle)
                if port:
                    k.CloseHandle(port)
            result = {"direct": direct, "iocp": async_io, "depth": depth,
                      "block_bytes": 4096, "batches": len(latencies),
                      "p50_ms": percentile(latencies, 0.5),
                      "p95_ms": percentile(latencies, 0.95),
                      "p99_ms": percentile(latencies, 0.99),
                      "iops": depth * len(latencies) * 1000 / sum(latencies),
                      "checksum": checksum}
            results.append(result)
            print(json.dumps(result), flush=True)
        # Measure per-query handle/port creation overhead separately from reads.
        latencies = []
        for _ in range(256):
            start = time.perf_counter_ns()
            handle = open_file(True, True)
            port = k.CreateIoCompletionPort(handle, None, 0, 1)
            if not port:
                k.CloseHandle(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            k.CloseHandle(handle)
            k.CloseHandle(port)
            latencies.append((time.perf_counter_ns() - start) / 1e6)
        results.append({"operation": "open-close-file-and-iocp-port-without-reading",
                        "p50_ms": percentile(latencies, 0.5),
                        "p95_ms": percentile(latencies, 0.95)})
    finally:
        k.VirtualFree(buffer, 0, 0x8000)
    report = {"file": filename, "file_bytes": args.file.stat().st_size,
              "read_only": True, "buffer_alignment": 4096, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

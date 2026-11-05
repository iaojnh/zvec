# Comparable cross-platform storage IOPS

`DiskANN Cross-platform Storage IOPS` builds this standalone program without
building Zvec. Native runners cover Linux, Windows, and macOS on x86_64/ARM64;
Android x86_64 and iOS ARM64 run inside their existing CI simulators. Windows
ARM64 additionally measures the verified blank local NVMe scratch volume when
available. The same PR retains the earlier index benchmarks and I/O diagnosis.

Each environment uses a deterministic, nonzero 256 MiB fixture, 4 KiB random
reads, ten seconds per case, and one/eight worker threads. Every worker has a
separate descriptor/handle. Every successful read checks its first 64-bit word
against the fixture generator. IOPS counts all completed operations; latency
samples are collected every 64 operations. Eight threads means up to eight
concurrent synchronous reads, rather than a single asynchronous queue of depth
eight. These results should be compared with each other; earlier fio/io_uring
and IOCP probes used different APIs, durations, or fixture sizes.

Two cache policies are reported separately:

- **Buffered warm:** read the entire fixture sequentially before timing.
- **Native no-cache:** `O_DIRECT` on Linux/Android, `FILE_FLAG_NO_BUFFERING` on
  Windows, and `F_NOCACHE` plus disabled read-ahead on macOS/iOS. Apple flags
  are advisory. The driver attempts `purge` before Apple no-cache cases only
  inside an ephemeral GitHub-hosted VM, and retains its return status.

Linux `/proc/self/io` and macOS `proc_pid_rusage` disk-byte accounting are
recorded to help check cache effects. Windows process transfer counters are
logical I/O counters. The public iOS SDK has no equivalent counter used here.
No-cache flags do not guarantee bypassing hypervisor or storage-controller
caches. Android O_DIRECT bypasses guest caching; emulator host caching may
remain. Mobile simulator results are not measurements of physical-phone flash.

The `storage-iops-summary` artifact contains a comparison table and full JSON;
each job retains per-case JSON, native logs, cache-command results, and host
metadata. Unsupported no-cache modes are labeled, never relabeled as uncached
buffered reads. CPU, filesystem, and virtual storage differences remain part of
the measured CI environment, not operating-system performance guarantees.

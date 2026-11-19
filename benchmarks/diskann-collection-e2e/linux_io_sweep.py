"""Pair DiskANN Collection QPS with cgroup block-read IOPS on Linux."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys


LIMITS = (2000, 4000, 8000, None)
CYCLE_COUNT = 3
RESULT_PREFIX = "ZVEC_COLLECTION_BENCH_RESULT "


def device_for(collection):
    device = collection.stat().st_dev
    sys_device = (Path("/sys/dev/block") / f"{os.major(device)}:{os.minor(device)}").resolve(strict=True)
    if (sys_device / "partition").exists():
        sys_device = sys_device.parent
    if any((sys_device / "slaves").iterdir()):
        raise RuntimeError(f"Expected an unstacked block device for {collection}")
    return (sys_device / "dev").read_text().strip()


def keyed_stats(path):
    stats = {}
    for line in path.read_text().splitlines():
        key, *fields = line.split()
        stats[key] = dict(field.split("=", 1) for field in fields)
    return stats


def direct_index_files(collection):
    expected = set(collection.rglob("embedding.index.*.proxima"))
    if not expected:
        raise RuntimeError(f"No persisted embedding index in {collection}")

    by_inode = {}
    for index in expected:
        stat = index.stat()
        by_inode[stat.st_dev, stat.st_ino] = index

    direct = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            stat = fd.stat()
            target = by_inode.get((stat.st_dev, stat.st_ino))
            if target is None:
                continue
            flags = next(
                line.split()[1]
                for line in (Path("/proc/self/fdinfo") / fd.name).read_text().splitlines()
                if line.startswith("flags:")
            )
        except FileNotFoundError:
            continue
        if int(flags, 8) & os.O_DIRECT:
            direct.add(target)
    if missing := expected - direct:
        raise RuntimeError("Index O_DIRECT descriptors missing: " + ", ".join(map(str, sorted(missing))))
    return [str(path.relative_to(collection)) for path in sorted(direct)]


class QueryBlockIO:
    def __init__(self, collection):
        collection = collection.resolve(strict=True)
        self.device = device_for(collection)
        self.index_files = direct_index_files(collection)
        unified = next((
            line.removeprefix("0::") for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ), None)
        if not unified or unified == "/":
            raise RuntimeError("Query block I/O requires a dedicated cgroup v2 scope")
        self.cgroup = Path("/sys/fs/cgroup") / unified.lstrip("/")
        self.stats = self.cgroup / "io.stat"
        limit = keyed_stats(self.cgroup / "io.max").get(self.device, {}).get("riops", "max")
        self.limit = None if limit == "max" else int(limit)

    def snapshot(self):
        row = keyed_stats(self.stats).get(self.device, {})
        return int(row.get("rios", 0)), int(row.get("rbytes", 0))

    def result(self, before, window_ns, queries):
        after = self.snapshot()
        ios = after[0] - before[0]
        read_bytes = after[1] - before[1]
        if ios <= 0 or read_bytes <= 0 or window_ns <= 0 or queries <= 0:
            raise RuntimeError(f"No accounted query block reads for {self.device}")
        return dict(
            accounting="Linux cgroup v2 io.stat",
            device=self.device,
            direct_index_files=self.index_files,
            limit_read_iops=self.limit,
            query_count=queries,
            elapsed_ns=window_ns,
            read_ios=ios,
            read_bytes=read_bytes,
            paired_qps=queries * 1e9 / window_ns,
            read_iops=ios * 1e9 / window_ns,
            read_ios_per_query=ios / queries,
        )


def run_query(data, collection, device, limit, log):
    command = ["sudo", "-n", "systemd-run", "--scope", "--quiet", "--property=IOAccounting=yes"]
    if limit is not None:
        command.append(f"--property=IOReadIOPSMax=/dev/block/{device} {limit}")
    command += [
        "--", "/usr/bin/setpriv",
        "--reuid", str(os.getuid()), "--regid", str(os.getgid()), "--init-groups", "--reset-env",
        "--", sys.executable, str(Path(__file__).with_name("python_benchmark.py").resolve()),
        "query", "diskann", str(data), str(collection), "--linux-block-io",
    ]
    print(f"DiskANN query, read IOPS limit {limit or 'max'}: {log.name}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    text = log.read_text(encoding="utf-8", errors="replace")
    results = [line.removeprefix(RESULT_PREFIX) for line in text.splitlines()
               if line.startswith(RESULT_PREFIX)]
    if status.returncode or len(results) != 1:
        raise RuntimeError(f"Query failed with code {status.returncode}:\n{text[-6000:]}")
    result = json.loads(results[0])
    io = result["linux_block_io"]
    if io["device"] != device or io["limit_read_iops"] != limit:
        raise RuntimeError(f"Unexpected cgroup device or read IOPS limit in {log}")
    return result


def pooled(trials):
    windows = [trial["linux_block_io"] for trial in trials]
    seconds = sum(run["elapsed_ns"] for run in windows) / 1e9
    queries = sum(run["query_count"] for run in windows)
    reads = sum(run["read_ios"] for run in windows)
    return dict(
        limit_read_iops=windows[0]["limit_read_iops"],
        trials=len(trials),
        queries=queries,
        seconds=seconds,
        read_ios=reads,
        read_bytes=sum(run["read_bytes"] for run in windows),
        qps=queries / seconds,
        read_iops=reads / seconds,
        read_ios_per_query=reads / queries,
        median_p95_ms=median(trial["p95_ms"] for trial in trials),
        median_recall_at_10=median(trial["recall_at_10"] for trial in trials),
    )


def markdown(label, budget, rows):
    lines = [
        f"## DiskANN direct-read IOPS and QPS ({label})",
        "",
        f"Zvec 0.8.1 Python Collection, SIFT-100K, one query worker, list_size={budget}.",
        "Rates use the same query wall-clock windows and Linux cgroup block-read counters.",
        "",
        "| Read IOPS cap | Measured read IOPS | QPS | Block reads/query | Median P95 (ms) | Recall@10 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['limit_read_iops'] or 'Uncapped'} | {row['read_iops']:.1f} | {row['qps']:.1f} | "
            f"{row['read_ios_per_query']:.2f} | {row['median_p95_ms']:.3f} | {row['median_recall_at_10']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("The IOPS sweep requires Linux cgroup v2")

    data = args.data.resolve(strict=True)
    collection = args.collection.resolve(strict=True)
    device = device_for(collection)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {limit: [] for limit in LIMITS}
    for cycle in range(CYCLE_COUNT):
        limits = LIMITS if cycle % 2 == 0 else reversed(LIMITS)
        for limit in limits:
            log = args.output / f"io-{limit or 'uncapped'}-{cycle + 1}.log"
            record = run_query(data, collection, device, limit, log)
            runs[limit].append(record)
            log.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    all_trials = [trial for group in runs.values() for trial in group]
    budgets = {trial["budget"] for trial in all_trials}
    if len(budgets) != 1 or min(trial["recall_at_10"] for trial in all_trials) < 0.93:
        raise RuntimeError("IOPS sweep needs the same query budget and valid recall in all trials")
    budget = budgets.pop()
    rows = [pooled(runs[limit]) for limit in LIMITS]
    report = dict(
        label=args.label,
        revision=os.environ.get("GITHUB_SHA"),
        device=device,
        budget=budget,
        rows=rows,
        raw_trials=all_trials,
    )
    (args.output / "io-sweep.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = markdown(args.label, budget, rows)
    (args.output / "io-sweep.md").write_text(summary, encoding="utf-8")
    print(summary)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(step_summary).open("a", encoding="utf-8") as stream:
            stream.write(summary)


if __name__ == "__main__":
    main()

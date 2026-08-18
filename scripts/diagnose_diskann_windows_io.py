"""Capture and replay the real Windows DiskANN I/O request stream."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build the Windows DiskANN diagnostic tools, capture the real "
            "search I/O trace, then compare its used IOContext with a fresh "
            "IOContext in the same process."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--capture-seconds", type=int, default=3)
    parser.add_argument("--replay-seconds", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=int, default=2)
    parser.add_argument("--queue-depths", nargs="+", type=int, default=(1, 2, 4, 8, 20))
    parser.add_argument(
        "--batch-gaps-us", nargs="+", type=int, default=(0, 50, 100, 250, 500, 1000)
    )
    parser.add_argument("--list-size", type=int, default=100)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-nodes",
        type=int,
        help="cache this many BFS nodes (legacy sizing mode)",
    )
    cache_group.add_argument(
        "--cache-budget-mb",
        type=int,
        help=(
            "size each physical index's BFS node cache from this nominal "
            "memory budget; "
            "recommended for cross-platform comparisons"
        ),
    )
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--top-k", default="50")
    parser.add_argument("--max-trace-records", type=int, default=1_000_000)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also rerun the older search and standalone I/O experiments",
    )
    args = parser.parse_args()
    if args.cache_nodes is None and args.cache_budget_mb is None:
        args.cache_nodes = 10_000
    return args


def write_console(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def run_logged(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> list[str]:
    rendered = [str(item) for item in command]
    write_console(f"> {subprocess.list2cmdline(rendered)}")
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8", newline="") as output:
        process = subprocess.Popen(
            rendered,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture process output")
        for line in process.stdout:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line.rstrip("\r\n"))
        return_code = process.wait()
    if return_code != 0:
        msg = f"Command failed with exit code {return_code}: {rendered[0]}"
        raise RuntimeError(msg)
    return lines


def executable(name: str, repo_root: Path) -> str:
    located = shutil.which(name)
    if located:
        return located
    candidate = repo_root / ".venv" / "Scripts" / f"{name}.exe"
    if candidate.is_file():
        return str(candidate)
    msg = f"Cannot find executable: {name}"
    raise FileNotFoundError(msg)


def find_tool(build_dir: Path, name: str) -> Path | None:
    filename = f"{name}.exe"
    candidates = (
        build_dir / "bin" / filename,
        build_dir / "bin" / "Release" / filename,
        build_dir / "Release" / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def ensure_tools(
    repo_root: Path,
    build_dir: Path,
    output_dir: Path,
    *,
    parallel: int,
    skip_build: bool,
    full: bool,
) -> tuple[Path, Path | None]:
    names = ("bench_original", "diskann_iocp_bench") if full else ("bench_original",)
    tools = {name: find_tool(build_dir, name) for name in names}
    if not skip_build:
        cmake = executable("cmake", repo_root)
        configure: list[str | Path] = [
            cmake,
            "-S",
            repo_root,
            "-B",
            build_dir,
            "-DBUILD_TOOLS=ON",
        ]
        if not (build_dir / "CMakeCache.txt").is_file():
            configure.extend(
                [
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_PYTHON_BINDINGS=OFF",
                    "-DBUILD_C_BINDINGS=OFF",
                    "-DBUILD_ZVEC_SHARED=OFF",
                    "-DBUILD_ZVEC_CORE_SHARED=OFF",
                    "-DBUILD_ZVEC_AILEGO_SHARED=OFF",
                ]
            )
        run_logged(
            configure,
            cwd=repo_root,
            log_path=output_dir / "cmake_configure.log",
        )
        run_logged(
            [
                cmake,
                "--build",
                build_dir,
                "--target",
                *names,
                "--config",
                "Release",
                "--parallel",
                str(parallel),
            ],
            cwd=repo_root,
            log_path=output_dir / "cmake_build.log",
        )
        tools = {name: find_tool(build_dir, name) for name in names}

    missing = [name for name, path in tools.items() if path is None]
    if missing:
        msg = f"Missing tools in {build_dir}: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    bench_tool = tools["bench_original"]
    iocp_tool = tools.get("diskann_iocp_bench")
    if bench_tool is None or (full and iocp_tool is None):
        raise AssertionError("Tool discovery and validation disagree")
    return bench_tool, iocp_tool


def yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_search_config(
    path: Path,
    *,
    index_file: Path,
    query_file: Path,
    capture_seconds: int,
    list_size: int,
    cache_nodes: int | None,
    cache_budget_mb: int | None,
    beam_size: int,
    top_k: str,
) -> None:
    if cache_budget_mb is not None:
        cache_setting = (
            "    zvec.diskann.searcher.cache_node_budget_bytes: !!int "
            f"{cache_budget_mb * 1024 * 1024}"
        )
    else:
        cache_setting = (
            f"    zvec.diskann.searcher.cache_node_num: !!int {cache_nodes or 0}"
        )
    content = f"""SearcherCommon:
    SearcherClass: DiskAnnSearcher
    IndexPath: {yaml_string(index_file)}
    TopK: {yaml_string(top_k)}
    QueryFile: {yaml_string(query_file)}
    QueryType: float
    QueryFirstSep: ";"
    QuerySecondSep: " "
    BenchThreadCount: 1
    BenchSecs: {capture_seconds}
    BenchIterCount: 10000000
    CompareById: true
    ContainerType: FileReadStorage
    LogLevel: Info
SearcherParams:
{cache_setting}
    zvec.diskann.searcher.list_size: !!int {list_size}
    zvec.diskann.searcher.beam_size: !!int {beam_size}
ContainerParams: {{}}
"""
    path.write_text(content, encoding="utf-8")


def extract_rows(lines: Sequence[str]) -> list[dict[str, str]]:
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("mode,random_access_hint,")
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError("IOCP benchmark output did not contain a CSV table")
    table = [line for line in lines[header_index:] if line.count(",") >= 16]
    rows = list(csv.DictReader(table))
    required_fields = {"batch_gap_us", "actual_gap_us"}
    if not rows or not required_fields.issubset(rows[0]):
        raise RuntimeError(
            "IOCP benchmark does not expose gap diagnostics; rebuild without "
            "--skip-build"
        )
    return rows


def search_metrics(lines: Sequence[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    patterns = {
        "completion_mode": r"completion=([a-z_]+)",
        "qps": r"Avg latency: [0-9.]+ms qps: ([0-9.]+)",
        "avg_latency_ms": r"Avg latency: ([0-9.]+)ms qps:",
        "reads_per_query": r"reads/query=([0-9.]+)",
        "batches_per_query": r"batches/query=([0-9.]+)",
        "reads_per_batch": r"reads/batch=([0-9.]+)",
        "io_us_per_query": r"io_us/query=([0-9.]+)",
        "cpu_us_per_query": r"cpu_us/query=([0-9.]+)",
        "iocp_wait_us_per_query": r"iocp_wait_us/query=([0-9.]+)",
        "readfile_submit_us_per_query": r"readfile_submit_us/query=([0-9.]+)",
        "get_overlapped_us_per_query": r"get_overlapped_us/query=([0-9.]+)",
        "completions_per_dequeue": r"completions/dequeue=([0-9.]+)",
        "batch_submit_us": r"batch_submit_us=([0-9.]+)",
        "first_completion_us": r"first_completion_us=([0-9.]+)",
        "batch_duration_us": r"batch_duration_us=([0-9.]+)",
    }
    for line in lines:
        for name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                metrics[name] = match.group(1)
    return metrics


def context_replay_metrics(lines: Sequence[str]) -> dict[str, dict[str, str]]:
    patterns = {
        "phase": r"phase=([a-z_]+)",
        "context": r"context=([a-z_]+)",
        "batches": r"batches=([0-9]+)",
        "reads": r"reads=([0-9]+)",
        "reads_per_batch": r"reads/batch=([0-9.]+)",
        "iops": r"iops=([0-9.]+)",
        "pending_ratio_pct": r"pending_ratio=([0-9.]+)%",
        "max_outstanding": r"max_outstanding=([0-9]+)",
        "batch_submit_us": r"submit_us/batch=([0-9.]+)",
        "first_completion_us": r"first_completion_us=([0-9.]+)",
        "batch_duration_us": r"batch_duration_us=([0-9.]+)",
        "iocp_wait_us_per_batch": r"iocp_wait_us/batch=([0-9.]+)",
        "readfile_submit_us_per_read": r"readfile_submit_us/read=([0-9.]+)",
        "get_overlapped_us_per_read": r"get_overlapped_us/read=([0-9.]+)",
        "completions_per_dequeue": r"completions/dequeue=([0-9.]+)",
        "max_dequeued_once": r"max_dequeued_once=([0-9]+)",
        "buffer_bytes": r"buffer_bytes=([0-9]+)",
        "read_stride": r"read_stride=([0-9]+)",
    }
    replays: dict[str, dict[str, str]] = {}
    for line in lines:
        if "DiskAnn in-process IOCP replay:" not in line:
            continue
        metrics: dict[str, str] = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                metrics[name] = match.group(1)
        context = metrics.get("context")
        if context is not None:
            replays[context] = metrics
    required = {
        "phase",
        "context",
        "batches",
        "reads",
        "reads_per_batch",
        "iops",
        "pending_ratio_pct",
        "max_outstanding",
        "batch_submit_us",
        "first_completion_us",
        "batch_duration_us",
        "iocp_wait_us_per_batch",
        "readfile_submit_us_per_read",
        "get_overlapped_us_per_read",
        "completions_per_dequeue",
        "max_dequeued_once",
        "buffer_bytes",
        "read_stride",
    }
    if set(replays) != {"used", "fresh"} or any(
        not required.issubset(metrics) for metrics in replays.values()
    ):
        raise RuntimeError(
            "bench_original did not emit both used and fresh in-process "
            "context replay diagnostics; rebuild without --skip-build"
        )
    return replays


def add_live_search_context(
    context_replays: dict[str, dict[str, str]], live_search: dict[str, str]
) -> None:
    required = {
        "reads_per_batch",
        "batches_per_query",
        "iocp_wait_us_per_query",
        "batch_submit_us",
        "first_completion_us",
        "batch_duration_us",
        "completions_per_dequeue",
    }
    if not required.issubset(live_search):
        raise RuntimeError(
            "The context replay process did not retain the preceding live "
            "search diagnostics"
        )
    batches_per_query = float(live_search["batches_per_query"])
    if batches_per_query <= 0:
        raise RuntimeError("Live search reported no I/O batches")
    live_fields = {
        "live_search_reads_per_batch": live_search["reads_per_batch"],
        "live_search_batch_submit_us": live_search["batch_submit_us"],
        "live_search_first_completion_us": live_search["first_completion_us"],
        "live_search_batch_duration_us": live_search["batch_duration_us"],
        "live_search_iocp_wait_us_per_batch": (
            f"{float(live_search['iocp_wait_us_per_query']) / batches_per_query:.2f}"
        ),
        "live_search_completions_per_dequeue": live_search["completions_per_dequeue"],
    }
    for replay in context_replays.values():
        replay.update(live_fields)


def require_search_mode(metrics: dict[str, str], expected: str) -> None:
    actual = metrics.get("completion_mode")
    if actual == expected:
        return
    msg = (
        f"Expected DiskANN completion mode '{expected}', got "
        f"'{actual or 'no mode marker'}'. Rebuild bench_original without "
        "--skip-build so the drain-first diagnostic code is included."
    )
    raise RuntimeError(msg)


def best_row(rows: Sequence[dict[str, str]], mode: str) -> dict[str, str] | None:
    candidates = [row for row in rows if row["mode"] == mode]
    return max(candidates, key=lambda row: float(row["iops"]), default=None)


def gap_interpretation(
    rows: Sequence[dict[str, str]],
    baseline: dict[str, str],
    *,
    gapped_mode: str,
    label: str,
) -> list[str]:
    gapped_rows = [row for row in rows if row["mode"] == gapped_mode]
    baseline_batch_ms = float(baseline["batch_duration_ms"])
    if not gapped_rows or baseline_batch_ms <= 0.0:
        return []

    worst = max(gapped_rows, key=lambda row: float(row["batch_duration_ms"]))
    worst_batch_ms = float(worst["batch_duration_ms"])
    worst_ratio = worst_batch_ms / baseline_batch_ms
    worst_gap = int(worst["batch_gap_us"])
    worst_actual_gap = float(worst["actual_gap_us"])
    lines = [
        (
            f"- The largest {label} gapped batch duration is "
            f"**{worst_batch_ms:.2f} ms** at **{worst_gap} us** requested "
            f"gap (**{worst_actual_gap:.1f} us** actual), or "
            f"**{worst_ratio:.1f}x** the zero-gap replay."
        )
    ]
    material_rows = [
        row
        for row in gapped_rows
        if float(row["batch_duration_ms"]) / baseline_batch_ms >= 2.0
    ]
    if material_rows:
        first_material = min(material_rows, key=lambda row: int(row["batch_gap_us"]))
        first_material_gap = int(first_material["batch_gap_us"])
        first_material_actual_gap = float(first_material["actual_gap_us"])
    else:
        first_material_gap = None
        first_material_actual_gap = None
    if first_material_actual_gap is not None and first_material_actual_gap <= 300:
        lines.append(
            f"- The latency doubles by **{first_material_gap} us** requested "
            f"gap (**{first_material_actual_gap:.1f} us** actual); this "
            "matches the search's short dependency gaps and can explain its "
            "effective-QD collapse."
        )
    elif first_material_gap is not None:
        lines.append(
            f"- Latency only doubles at **{first_material_gap} us** requested "
            f"gap (**{first_material_actual_gap:.1f} us** actual), which "
            "is longer than the observed search dependency gap and is "
            "unlikely to be the primary cause."
        )
    else:
        lines.append(
            "- Short queue-idle gaps do not reproduce the full-search batch "
            "latency and do not explain the effective-QD collapse."
        )
    return lines


def random_batched_interpretation(
    random_batched: dict[str, str] | None,
    trace_batched: dict[str, str],
    search_metrics: dict[str, str],
) -> list[str]:
    if random_batched is None:
        return ["- Fresh-random batched replay was not produced."]

    random_batch_ms = float(random_batched["batch_duration_ms"])
    trace_batch_ms = float(trace_batched["batch_duration_ms"])
    trace_ratio = random_batch_ms / trace_batch_ms
    lines = [
        (
            f"- Fresh-random zero-gap batches take **{random_batch_ms:.2f} "
            f"ms**, or **{trace_ratio:.1f}x** repeated-trace batches."
        )
    ]
    search_batch_us = search_metrics.get("batch_duration_us")
    if search_batch_us is None:
        return lines

    search_batch_ms = float(search_batch_us) / 1000.0
    search_ratio = random_batch_ms / search_batch_ms
    if 0.7 <= search_ratio <= 1.3:
        lines.append(
            f"- Fresh-random batching reproduces **{search_ratio:.0%}** of "
            "full-search batch latency; repeated-offset cache warmth is the "
            "leading explanation."
        )
    elif search_ratio < 0.5:
        lines.append(
            f"- Fresh-random batching reproduces only **{search_ratio:.0%}** "
            "of full-search batch latency; address warmth does not explain "
            "the slowdown."
        )
    else:
        lines.append(
            f"- Fresh-random batching reaches **{search_ratio:.0%}** of "
            "full-search batch latency; repeat the run before assigning the "
            "remaining difference."
        )
    return lines


def reader_buffer_interpretation(
    rows: Sequence[dict[str, str]], search_metrics: dict[str, str]
) -> list[str]:
    mode_labels = {
        "reader_virtual_compact": "VirtualAlloc compact",
        "reader_virtual_context": "VirtualAlloc 512 KiB",
        "reader_aligned_compact": "_aligned_malloc compact",
        "reader_aligned_context": "_aligned_malloc 512 KiB",
    }
    reader_rows = {mode: best_row(rows, mode) for mode in mode_labels}
    missing = [mode for mode, row in reader_rows.items() if row is None]
    if missing:
        return [
            (
                "- Reader destination-buffer allocation matrix is incomplete; "
                "rebuild the IOCP benchmark without `--skip-build`."
            )
        ]

    measured = {mode: row for mode, row in reader_rows.items() if row is not None}
    compact = measured["reader_virtual_compact"]
    context = measured["reader_aligned_context"]
    compact_batch_ms = float(compact["batch_duration_ms"])
    context_batch_ms = float(context["batch_duration_ms"])
    allocation_ratio = context_batch_ms / compact_batch_ms
    slowest_mode, slowest = max(
        measured.items(), key=lambda item: float(item[1]["batch_duration_ms"])
    )
    fastest_mode, fastest = min(
        measured.items(), key=lambda item: float(item[1]["batch_duration_ms"])
    )
    spread = float(slowest["batch_duration_ms"]) / float(fastest["batch_duration_ms"])
    lines = [
        (
            "- The search-equivalent `_aligned_malloc` 512 KiB buffer takes "
            f"**{context_batch_ms:.2f} ms/batch**, or "
            f"**{allocation_ratio:.1f}x** the VirtualAlloc compact buffer "
            f"(**{compact_batch_ms:.2f} ms/batch**)."
        ),
        (
            f"- Across all four buffer variants, the slowest is "
            f"**{mode_labels[slowest_mode]}** and the fastest is "
            f"**{mode_labels[fastest_mode]}**; batch latency spread is "
            f"**{spread:.1f}x**."
        ),
    ]

    search_batch_us = search_metrics.get("batch_duration_us")
    if search_batch_us is None:
        return lines
    search_batch_ms = float(search_batch_us) / 1000.0
    search_ratio = context_batch_ms / search_batch_ms
    if 0.7 <= search_ratio <= 1.3:
        lines.append(
            f"- The search-equivalent buffer reproduces **{search_ratio:.0%}** "
            "of full-search batch latency; buffer allocation is the leading "
            "cause and should be changed in `DiskAnnContext`."
        )
    elif allocation_ratio >= 1.5:
        lines.append(
            f"- Buffer allocation is material but reproduces only "
            f"**{search_ratio:.0%}** of full-search batch latency; it explains "
            "part, but not all, of the loss."
        )
    else:
        lines.append(
            f"- The search-equivalent buffer reproduces only "
            f"**{search_ratio:.0%}** of full-search batch latency. Allocation "
            "API and reserved size do not explain the slowdown."
        )
    return lines


def context_replay_interpretation(
    context_replays: dict[str, dict[str, str]],
) -> list[str]:
    used = context_replays.get("used")
    fresh = context_replays.get("fresh")
    if used is None or fresh is None:
        return ["- The used-versus-fresh context comparison is incomplete."]

    used_iops = float(used["iops"])
    fresh_iops = float(fresh["iops"])
    used_batch_us = float(used["batch_duration_us"])
    fresh_batch_us = float(fresh["batch_duration_us"])
    iops_ratio = fresh_iops / used_iops
    latency_ratio = fresh_batch_us / used_batch_us
    lines = [
        (
            "- With the same process, thread, reader, requests, and buffer, a "
            f"fresh context reaches **{fresh_iops:.0f} IOPS** versus "
            f"**{used_iops:.0f} IOPS** for the used search context "
            f"(**{iops_ratio:.1f}x**); batch latency is "
            f"**{fresh_batch_us:.0f} us** versus **{used_batch_us:.0f} us**."
        )
    ]
    if iops_ratio >= 2.0 and latency_ratio <= 0.7:
        lines.append(
            "- **Conclusion:** recreating the IOContext removes the slowdown. "
            "The persistent state of the search file handle or completion "
            "port is the cause."
        )
    elif 0.7 <= iops_ratio <= 1.3 and 0.7 <= latency_ratio <= 1.3:
        lines.append(
            "- **Conclusion:** a fresh IOContext does not change performance. "
            "Persistent file-handle and completion-port state are ruled out."
        )
    else:
        lines.append(
            "- **Conclusion:** recreating the IOContext has a partial effect, "
            "so context state contributes but is not the only cause."
        )
    return lines


def interpretation(
    rows: Sequence[dict[str, str]],
    search_runs: dict[str, dict[str, str]],
    context_replays: dict[str, dict[str, str]],
) -> list[str]:
    uniform = best_row(rows, "uniform")
    continuous = best_row(rows, "trace_continuous")
    batched = best_row(rows, "trace_batched")
    reader = best_row(rows, "reader_virtual_compact")
    if uniform is None or continuous is None or batched is None or reader is None:
        return ["- Not enough replay modes were produced for comparison."]

    uniform_iops = float(uniform["iops"])
    continuous_iops = float(continuous["iops"])
    batched_iops = float(batched["iops"])
    reader_iops = float(reader["iops"])
    access_ratio = continuous_iops / uniform_iops
    scheduling_ratio = batched_iops / continuous_iops
    reader_ratio = reader_iops / batched_iops
    search_baseline = search_runs.get("interleaved", {})
    random_batched = best_row(rows, "random_batched")
    lines = [
        (
            f"- Real offsets with a continuously full queue retain "
            f"**{access_ratio:.1%}** of the uniform-random peak "
            f"({continuous_iops:.0f} vs {uniform_iops:.0f} IOPS)."
        ),
        (
            f"- Preserving DiskANN's original batch barriers retains "
            f"**{scheduling_ratio:.1%}** of continuous trace replay "
            f"({batched_iops:.0f} vs {continuous_iops:.0f} IOPS)."
        ),
        (
            f"- The project's `WindowsAlignedFileReader` retains "
            f"**{reader_ratio:.1%}** of direct batched IOCP replay "
            f"({reader_iops:.0f} vs {batched_iops:.0f} IOPS)."
        ),
    ]
    lines.extend(
        gap_interpretation(
            rows, batched, gapped_mode="trace_gapped", label="repeated-trace"
        )
    )
    lines.extend(
        random_batched_interpretation(random_batched, batched, search_baseline)
    )
    lines.extend(reader_buffer_interpretation(rows, search_baseline))
    lines.extend(context_replay_interpretation(context_replays))
    if random_batched is not None:
        lines.extend(
            gap_interpretation(
                rows,
                random_batched,
                gapped_mode="random_gapped",
                label="fresh-random",
            )
        )
    if access_ratio < 0.7:
        lines.append(
            "- The captured offset/size pattern itself has a substantial cost."
        )
    if scheduling_ratio < 0.7:
        lines.append(
            "- Batch barriers leave significant device parallelism unused; "
            "cross-batch pipelining is the strongest next optimization target."
        )
    if reader_ratio < 0.7:
        lines.append(
            "- The performance loss is reproduced inside the project reader; "
            "focus on its submit/completion path and the timing columns below."
        )
    drain_first = search_runs.get("drain_first", {})
    if search_baseline.get("qps") and drain_first.get("qps"):
        baseline_qps = float(search_baseline["qps"])
        drain_qps = float(drain_first["qps"])
        drain_ratio = drain_qps / baseline_qps
        lines.append(
            f"- Drain-first search reaches **{drain_qps:.1f} QPS** versus "
            f"**{baseline_qps:.1f} QPS** for interleaved completion "
            f"(**{drain_ratio:.1%}**)."
        )
        if drain_ratio >= 1.1:
            lines.append(
                "- Draining the IOCP batch before node processing materially "
                "improves search throughput; completion/compute interleaving "
                "is a viable Windows optimization target."
            )
        elif drain_ratio <= 0.9:
            lines.append(
                "- Drain-first is materially slower; retaining I/O and CPU "
                "overlap is preferable on this system."
            )
        else:
            lines.append(
                "- Drain-first does not materially change search throughput; "
                "completion/compute interleaving is not the primary loss."
            )
    if search_baseline.get("qps") and search_baseline.get("reads_per_query"):
        search_read_iops = float(search_baseline["qps"]) * float(
            search_baseline["reads_per_query"]
        )
        search_ratio = search_read_iops / reader_iops
        lines.append(
            f"- Full search delivers about **{search_read_iops:.0f} read IOPS**, "
            f"or **{search_ratio:.1%}** of reader replay; the remaining loss is "
            "in search orchestration rather than the reader wrapper."
        )
    if access_ratio >= 0.7 and scheduling_ratio >= 0.7 and reader_ratio >= 0.7:
        lines.append(
            "- Raw IOCP, real offsets, batch barriers, and the project reader "
            "all retain good throughput."
        )
    return lines


CONTEXT_FIELDS = (
    "phase",
    "context",
    "batches",
    "reads",
    "reads_per_batch",
    "iops",
    "pending_ratio_pct",
    "max_outstanding",
    "batch_submit_us",
    "first_completion_us",
    "batch_duration_us",
    "iocp_wait_us_per_batch",
    "readfile_submit_us_per_read",
    "get_overlapped_us_per_read",
    "completions_per_dequeue",
    "max_dequeued_once",
    "buffer_bytes",
    "read_stride",
    "live_search_reads_per_batch",
    "live_search_batch_submit_us",
    "live_search_first_completion_us",
    "live_search_batch_duration_us",
    "live_search_iocp_wait_us_per_batch",
    "live_search_completions_per_dequeue",
)


def write_context_csv(
    output_dir: Path, context_replays: dict[str, dict[str, str]]
) -> None:
    with (output_dir / "context_replay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=CONTEXT_FIELDS)
        writer.writeheader()
        for context in ("used", "fresh"):
            replay = context_replays[context]
            writer.writerow({name: replay.get(name, "") for name in CONTEXT_FIELDS})


def context_report_lines(
    context_replays: dict[str, dict[str, str]],
) -> list[str]:
    used_context = context_replays["used"]
    lines = [
        "## In-process context replay",
        "",
        (
            "| Context | Batches | Reads/batch | IOPS | First us | Batch us | "
            "IOCP wait us/batch | Submit us/batch | Comp/dequeue |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| live_search |  | {live_search_reads_per_batch} |  | "
            "{live_search_first_completion_us} | "
            "{live_search_batch_duration_us} | "
            "{live_search_iocp_wait_us_per_batch} | "
            "{live_search_batch_submit_us} | "
            "{live_search_completions_per_dequeue} |"
        ).format(**used_context),
    ]
    for context in ("used", "fresh"):
        lines.append(
            (
                "| {context} | {batches} | {reads_per_batch} | {iops} | "
                "{first_completion_us} | {batch_duration_us} | "
                "{iocp_wait_us_per_batch} | {batch_submit_us} | "
                "{completions_per_dequeue} |"
            ).format(**context_replays[context])
        )
    lines.extend(
        [
            "",
            (
                "Both controls run after the first real query. They use the "
                "same process, thread, reader, request batches, and sector "
                "buffer; only the IOContext and its file handle/completion "
                "port change."
            ),
        ]
    )
    return lines


def write_context_results(
    output_dir: Path, context_replays: dict[str, dict[str, str]]
) -> None:
    write_context_csv(output_dir, context_replays)
    summary = "\n".join(
        [
            "# DiskANN Windows IOContext diagnosis",
            "",
            *context_report_lines(context_replays),
            "",
            "## Interpretation",
            "",
            *context_replay_interpretation(context_replays),
            "",
        ]
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def write_results(
    output_dir: Path,
    rows: list[dict[str, str]],
    search_runs: dict[str, dict[str, str]],
    context_replays: dict[str, dict[str, str]],
) -> None:
    if not rows:
        raise RuntimeError("No replay results were produced")
    fields = list(rows[0])
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        (
            "| Mode | QD | Max batch | Gap us | Actual gap us | IOPS | Avg ms | "
            "P95 ms | Effective QD | First ms | Batch ms | IOCP wait ms | "
            "ReadFile us | GetOverlapped us | Comp/dequeue |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in rows:
        table_lines.append(
            "| {mode} | {queue_depth} | {max_batch_size} | {batch_gap_us} | "
            "{actual_gap_us} | {iops} | "
            "{avg_latency_ms} | {p95_latency_ms} | {effective_qd} | "
            "{first_completion_ms} | {batch_duration_ms} | "
            "{iocp_wait_ms_per_batch} | {readfile_submit_us_per_read} | "
            "{get_overlapped_us_per_read} | "
            "{completions_per_dequeue} |".format(**row)
        )
    search_fields = [
        "mode",
        "qps",
        "avg_latency_ms",
        "reads_per_query",
        "batches_per_query",
        "io_us_per_query",
        "cpu_us_per_query",
        "iocp_wait_us_per_query",
        "batch_submit_us",
        "first_completion_us",
        "batch_duration_us",
        "readfile_submit_us_per_query",
        "get_overlapped_us_per_query",
        "completions_per_dequeue",
    ]
    with (output_dir / "search_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=search_fields)
        writer.writeheader()
        for mode, metrics in search_runs.items():
            writer.writerow(
                {
                    name: mode if name == "mode" else metrics.get(name, "")
                    for name in search_fields
                }
            )

    search_lines = [
        "## DiskANN search A/B",
        "",
        (
            "| Mode | QPS | Avg ms | Reads/query | Batches/query | I/O us/query | "
            "CPU us/query | IOCP wait us/query | Batch submit us | First us | "
            "Batch duration us | Comp/dequeue |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]
    for mode in ("interleaved", "drain_first"):
        metrics = search_runs.get(mode, {})
        search_lines.append(
            "| {mode} | {qps} | {avg_latency_ms} | {reads_per_query} | "
            "{batches_per_query} | {io_us_per_query} | {cpu_us_per_query} | "
            "{iocp_wait_us_per_query} | {batch_submit_us} | "
            "{first_completion_us} | {batch_duration_us} | "
            "{completions_per_dequeue} |".format(
                mode=mode,
                **{
                    name: metrics.get(name, "")
                    for name in search_fields
                    if name != "mode"
                },
            )
        )

    write_context_csv(output_dir, context_replays)
    context_lines = context_report_lines(context_replays)

    summary = "\n".join(
        [
            "# DiskANN Windows I/O trace diagnosis",
            "",
            *search_lines,
            "",
            (
                "The trace-capture run is separate from this A/B table. Both "
                "search measurements run without trace recording overhead."
            ),
            "",
            *context_lines,
            "",
            "## I/O replay",
            "",
            *table_lines,
            "",
            "- `uniform`: uniform random reads from the same index file.",
            "- `trace_continuous`: real offsets replayed with a full queue.",
            "- `trace_batched`: real offsets replayed with original barriers.",
            (
                "- `trace_gapped`: the same barriers plus a CPU-busy delay "
                "between batches; `Gap us` is excluded from batch latency."
            ),
            (
                "- `random_batched`: original batch sizes and read lengths "
                "with freshly generated random aligned offsets."
            ),
            (
                "- `random_gapped`: fresh-random batches plus the requested "
                "CPU-busy inter-batch delay."
            ),
            (
                "- `reader_virtual_compact` / `reader_virtual_context`: the "
                "same batches replayed through `WindowsAlignedFileReader` "
                "using `VirtualAlloc`, with a compact buffer or the 512 KiB "
                "DiskANN context size."
            ),
            (
                "- `reader_aligned_compact` / `reader_aligned_context`: the "
                "same reader replay using `_aligned_malloc`; the context "
                "variant matches DiskANN's search allocator and size."
            ),
            "",
            "## Interpretation",
            "",
            *interpretation(rows, search_runs, context_replays),
            "",
            (
                "The replay excludes DiskANN distance computation and cache "
                "traversal, so it isolates only the storage request pattern "
                "and scheduling barriers."
            ),
            (
                "`trace_batched` preserves batch boundaries but does not "
                "reproduce node processing between completion dequeues."
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if sys.platform != "win32":
        raise RuntimeError("This diagnostic script must run on Windows")
    for path, name in (
        (args.index_file, "--index-file"),
        (args.query_file, "--query-file"),
    ):
        if not path.is_file():
            msg = f"{name} does not exist: {path}"
            raise FileNotFoundError(msg)
    positive_values = (
        args.parallel,
        args.capture_seconds,
        args.replay_seconds,
        args.list_size,
        args.beam_size,
        args.max_trace_records,
        *args.queue_depths,
    )
    if any(value <= 0 for value in positive_values) or args.warmup_seconds < 0:
        raise ValueError("Durations, queue depths, and sizes must be positive")
    if any(value < 0 or value > 1_000_000 for value in args.batch_gaps_us):
        raise ValueError("--batch-gaps-us must be between 0 and 1000000")
    if 0 not in args.batch_gaps_us:
        raise ValueError("--batch-gaps-us must include 0 for the baseline")
    if args.cache_nodes is not None and not 0 <= args.cache_nodes <= 0xFFFFFFFF:
        raise ValueError("--cache-nodes must be between 0 and UINT32_MAX")
    if args.cache_budget_mb is not None and args.cache_budget_mb <= 0:
        raise ValueError("--cache-budget-mb must be positive")
    if args.cache_budget_mb is not None and args.cache_budget_mb > ((1 << 63) - 1) // (
        1024 * 1024
    ):
        raise ValueError("--cache-budget-mb exceeds INT64_MAX bytes")


def search_environment(
    *,
    drain_first: bool,
    trace_path: Path | None = None,
    max_trace_records: int = 0,
    context_replay_trace: Path | None = None,
    context_replay_seconds: int = 0,
    context_replay_warmup_seconds: int = 0,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ZVEC_DISKANN_IO_DIAGNOSTICS"] = "1"
    env["ZVEC_DISKANN_IO_PIPELINE"] = "0"
    env["ZVEC_DISKANN_IO_DRAIN_FIRST"] = "1" if drain_first else "0"
    env.pop("ZVEC_DISKANN_IO_TRACE", None)
    env.pop("ZVEC_DISKANN_IO_TRACE_MAX_RECORDS", None)
    env.pop("ZVEC_DISKANN_IO_CONTEXT_REPLAY", None)
    env.pop("ZVEC_DISKANN_IO_CONTEXT_REPLAY_SECONDS", None)
    env.pop("ZVEC_DISKANN_IO_CONTEXT_REPLAY_WARMUP_SECONDS", None)
    if trace_path is not None:
        env["ZVEC_DISKANN_IO_TRACE"] = str(trace_path)
        env["ZVEC_DISKANN_IO_TRACE_MAX_RECORDS"] = str(max_trace_records)
    if context_replay_trace is not None:
        env["ZVEC_DISKANN_IO_CONTEXT_REPLAY"] = str(context_replay_trace)
        env["ZVEC_DISKANN_IO_CONTEXT_REPLAY_SECONDS"] = str(context_replay_seconds)
        env["ZVEC_DISKANN_IO_CONTEXT_REPLAY_WARMUP_SECONDS"] = str(
            context_replay_warmup_seconds
        )
    return env


def run_context_comparison(
    bench_tool: Path,
    config_path: Path,
    output_dir: Path,
    trace_path: Path,
    *,
    replay_seconds: int,
    warmup_seconds: int,
) -> dict[str, dict[str, str]]:
    write_console("\nComparing the used and fresh IOContext...")
    lines = run_logged(
        [bench_tool, config_path],
        cwd=output_dir,
        env=search_environment(
            drain_first=False,
            context_replay_trace=trace_path,
            context_replay_seconds=replay_seconds,
            context_replay_warmup_seconds=warmup_seconds,
        ),
        log_path=output_dir / "context_replay.log",
    )
    context_metrics = context_replay_metrics(lines)
    live_search = search_metrics(lines)
    require_search_mode(live_search, "interleaved")
    add_live_search_context(context_metrics, live_search)
    return context_metrics


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    build_dir = (args.build_dir or repo_root / "build").resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir or build_dir / "diskann_iocp_diagnostics" / timestamp
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_tool, iocp_tool = ensure_tools(
        repo_root,
        build_dir,
        output_dir,
        parallel=args.parallel,
        skip_build=args.skip_build,
        full=args.full,
    )
    config_path = output_dir / "capture.yaml"
    trace_path = output_dir / "diskann_io_trace.csv"
    write_search_config(
        config_path,
        index_file=args.index_file.resolve(),
        query_file=args.query_file.resolve(),
        capture_seconds=args.capture_seconds,
        list_size=args.list_size,
        cache_nodes=args.cache_nodes,
        cache_budget_mb=args.cache_budget_mb,
        beam_size=args.beam_size,
        top_k=args.top_k,
    )

    write_console("\nCapturing real DiskANN I/O offsets...")
    run_logged(
        [bench_tool, config_path],
        cwd=output_dir,
        env=search_environment(
            drain_first=False,
            trace_path=trace_path,
            max_trace_records=args.max_trace_records,
        ),
        log_path=output_dir / "trace_capture.log",
    )
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise RuntimeError("DiskANN did not produce an I/O trace")

    context_metrics = run_context_comparison(
        bench_tool,
        config_path,
        output_dir,
        trace_path,
        replay_seconds=args.replay_seconds,
        warmup_seconds=args.warmup_seconds,
    )

    if not args.full:
        write_context_results(output_dir, context_metrics)
        write_console("\nDiagnosis complete.")
        write_console(f"Summary: {output_dir / 'summary.md'}")
        write_console(f"Context: {output_dir / 'context_replay.csv'}")
        write_console(f"Trace:   {trace_path}")
        return 0

    if iocp_tool is None:
        raise AssertionError("Full diagnostics require diskann_iocp_bench")

    write_console("\nRunning interleaved-completion search baseline...")
    interleaved_lines = run_logged(
        [bench_tool, config_path],
        cwd=output_dir,
        env=search_environment(drain_first=False),
        log_path=output_dir / "search_interleaved.log",
    )
    write_console("\nRunning drain-first search experiment...")
    drain_first_lines = run_logged(
        [bench_tool, config_path],
        cwd=output_dir,
        env=search_environment(drain_first=True),
        log_path=output_dir / "search_drain_first.log",
    )
    queue_depths = ",".join(str(value) for value in args.queue_depths)
    batch_gaps_us = ",".join(str(value) for value in args.batch_gaps_us)
    common: list[str | Path] = [
        iocp_tool,
        "--file",
        args.index_file.resolve(),
        "--queue-depths",
        queue_depths,
        "--warmup",
        str(args.warmup_seconds),
        "--duration",
        str(args.replay_seconds),
        "--random-access-hint",
        "on",
    ]

    write_console("\nRunning uniform-random IOCP baseline...")
    uniform_lines = run_logged(
        common,
        cwd=output_dir,
        log_path=output_dir / "uniform_replay.log",
    )
    write_console("\nRunning repeated-trace and fresh-random batch replays...")
    trace_lines = run_logged(
        [
            *common,
            "--trace-file",
            trace_path,
            "--trace-mode",
            "both",
            "--batch-gaps-us",
            batch_gaps_us,
            "--random-batched",
            "--reader-replay",
        ],
        cwd=output_dir,
        log_path=output_dir / "trace_replay.log",
    )
    rows = [*extract_rows(uniform_lines), *extract_rows(trace_lines)]
    interleaved_metrics = search_metrics(interleaved_lines)
    drain_first_metrics = search_metrics(drain_first_lines)
    require_search_mode(interleaved_metrics, "interleaved")
    require_search_mode(drain_first_metrics, "drain_first")
    search_runs = {
        "interleaved": interleaved_metrics,
        "drain_first": drain_first_metrics,
    }
    write_results(output_dir, rows, search_runs, context_metrics)

    write_console("\nDiagnosis complete.")
    write_console(f"Summary: {output_dir / 'summary.md'}")
    write_console(f"Results: {output_dir / 'results.csv'}")
    write_console(f"Search:  {output_dir / 'search_results.csv'}")
    write_console(f"Context: {output_dir / 'context_replay.csv'}")
    write_console(f"Trace:   {trace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        write_console("\nInterrupted.")
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"\nERROR: {error}\n")
        raise SystemExit(1) from None

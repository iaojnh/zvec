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
            "search I/O trace, and replay it with continuous and original "
            "batch scheduling."
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
    parser.add_argument("--list-size", type=int, default=100)
    parser.add_argument("--cache-nodes", type=int, default=10_000)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--top-k", default="50")
    parser.add_argument("--max-trace-records", type=int, default=1_000_000)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


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
) -> tuple[Path, Path]:
    names = ("bench_original", "diskann_iocp_bench")
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
    iocp_tool = tools["diskann_iocp_bench"]
    if bench_tool is None or iocp_tool is None:
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
    cache_nodes: int,
    beam_size: int,
    top_k: str,
) -> None:
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
    zvec.diskann.searcher.cache_node_num: !!int {cache_nodes}
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
    return list(csv.DictReader(table))


def search_metrics(lines: Sequence[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    patterns = {
        "qps": r"Avg latency: [0-9.]+ms qps: ([0-9.]+)",
        "avg_latency_ms": r"Avg latency: ([0-9.]+)ms qps:",
        "reads_per_query": r"reads/query=([0-9.]+)",
        "io_us_per_query": r"io_us/query=([0-9.]+)",
        "cpu_us_per_query": r"cpu_us/query=([0-9.]+)",
        "readfile_submit_us_per_query": r"readfile_submit_us/query=([0-9.]+)",
        "get_overlapped_us_per_query": r"get_overlapped_us/query=([0-9.]+)",
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


def best_row(rows: Sequence[dict[str, str]], mode: str) -> dict[str, str] | None:
    candidates = [row for row in rows if row["mode"] == mode]
    return max(candidates, key=lambda row: float(row["iops"]), default=None)


def interpretation(
    rows: Sequence[dict[str, str]], capture_metrics: dict[str, str]
) -> list[str]:
    uniform = best_row(rows, "uniform")
    continuous = best_row(rows, "trace_continuous")
    batched = best_row(rows, "trace_batched")
    reader = best_row(rows, "reader_batched")
    if uniform is None or continuous is None or batched is None or reader is None:
        return ["- Not enough replay modes were produced for comparison."]

    uniform_iops = float(uniform["iops"])
    continuous_iops = float(continuous["iops"])
    batched_iops = float(batched["iops"])
    reader_iops = float(reader["iops"])
    access_ratio = continuous_iops / uniform_iops
    scheduling_ratio = batched_iops / continuous_iops
    reader_ratio = reader_iops / batched_iops
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
    elif capture_metrics.get("qps") and capture_metrics.get("reads_per_query"):
        search_read_iops = float(capture_metrics["qps"]) * float(
            capture_metrics["reads_per_query"]
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


def write_results(
    output_dir: Path,
    rows: list[dict[str, str]],
    capture_metrics: dict[str, str],
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
            "| Mode | QD | Max batch | IOPS | Avg ms | P95 ms | Effective QD | "
            "First ms | Batch ms | IOCP wait ms | ReadFile us | "
            "GetOverlapped us | Comp/dequeue |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for row in rows:
        table_lines.append(
            "| {mode} | {queue_depth} | {max_batch_size} | {iops} | "
            "{avg_latency_ms} | {p95_latency_ms} | {effective_qd} | "
            "{first_completion_ms} | {batch_duration_ms} | "
            "{iocp_wait_ms_per_batch} | {readfile_submit_us_per_read} | "
            "{get_overlapped_us_per_read} | "
            "{completions_per_dequeue} |".format(**row)
        )
    search_lines = ["## Captured DiskANN search", ""]
    if capture_metrics:
        labels = {
            "qps": "QPS",
            "avg_latency_ms": "Average latency (ms)",
            "reads_per_query": "Reads/query",
            "io_us_per_query": "I/O time/query (us)",
            "cpu_us_per_query": "CPU time/query (us)",
            "readfile_submit_us_per_query": "ReadFile submit/query (us)",
            "get_overlapped_us_per_query": "GetOverlappedResult/query (us)",
            "batch_submit_us": "Batch submit (us)",
            "first_completion_us": "First completion/batch (us)",
            "batch_duration_us": "Batch duration (us)",
        }
        search_lines.extend(
            f"- {labels[name]}: {capture_metrics[name]}"
            for name in labels
            if name in capture_metrics
        )
    else:
        search_lines.append("- Search metrics were not found in the capture log.")

    summary = "\n".join(
        [
            "# DiskANN Windows I/O trace diagnosis",
            "",
            *search_lines,
            "",
            "## I/O replay",
            "",
            *table_lines,
            "",
            "- `uniform`: uniform random reads from the same index file.",
            "- `trace_continuous`: real offsets replayed with a full queue.",
            "- `trace_batched`: real offsets replayed with original barriers.",
            (
                "- `reader_batched`: the same batches replayed through the "
                "project's `WindowsAlignedFileReader`."
            ),
            "",
            "## Interpretation",
            "",
            *interpretation(rows, capture_metrics),
            "",
            (
                "The replay excludes DiskANN distance computation and cache "
                "traversal, so it isolates only the storage request pattern "
                "and scheduling barriers."
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
    if args.cache_nodes < 0:
        raise ValueError("--cache-nodes must not be negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo_root = args.repo_root.resolve()
    build_dir = (args.build_dir or repo_root / "build").resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        beam_size=args.beam_size,
        top_k=args.top_k,
    )

    capture_env = os.environ.copy()
    capture_env["ZVEC_DISKANN_IO_DIAGNOSTICS"] = "1"
    capture_env["ZVEC_DISKANN_IO_PIPELINE"] = "0"
    capture_env["ZVEC_DISKANN_IO_TRACE"] = str(trace_path)
    capture_env["ZVEC_DISKANN_IO_TRACE_MAX_RECORDS"] = str(args.max_trace_records)
    write_console("\nCapturing real DiskANN I/O offsets...")
    capture_lines = run_logged(
        [bench_tool, config_path],
        cwd=output_dir,
        env=capture_env,
        log_path=output_dir / "capture.log",
    )
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise RuntimeError("DiskANN did not produce an I/O trace")

    queue_depths = ",".join(str(value) for value in args.queue_depths)
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
    write_console("\nReplaying captured offsets...")
    trace_lines = run_logged(
        [
            *common,
            "--trace-file",
            trace_path,
            "--trace-mode",
            "both",
            "--reader-replay",
        ],
        cwd=output_dir,
        log_path=output_dir / "trace_replay.log",
    )
    rows = [*extract_rows(uniform_lines), *extract_rows(trace_lines)]
    write_results(output_dir, rows, search_metrics(capture_lines))

    write_console("\nDiagnosis complete.")
    write_console(f"Summary: {output_dir / 'summary.md'}")
    write_console(f"Results: {output_dir / 'results.csv'}")
    write_console(f"Trace:   {trace_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        write_console("\nInterrupted.")
        raise SystemExit(130) from None
    except Exception as error:
        sys.stderr.write(f"\nERROR: {error}\n")
        raise SystemExit(1) from None

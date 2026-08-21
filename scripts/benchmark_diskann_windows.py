r"""Run the DiskANN benchmark matrix on Windows.

The defaults intentionally match the supplied Cohere 1M Windows benchmark:

* FP32 and FP16 indexes
* list_size: 100, 300, 500
* query threads: 1, 2, 4
* 30 seconds per throughput run
* max degree 32, builder list size 50, PQ chunks 384
* 10,000 cached nodes

The script drives the repository's native C++ benchmark tools. It creates all
YAML files and writes raw logs,
CSV/JSON data, environment metadata, and document-ready Markdown tables under
D:/zvec-iaojnh/diskann_bench_windows/<timestamp>_<git-sha>.

Run from an "x64 Native Tools Command Prompt for VS 2022", with the project's
virtual environment activated:

    python scripts\benchmark_diskann_windows.py

If the generated ground-truth file does not exist, the script creates it from
the Cohere training/query files and reuses it on later runs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes
import datetime as dt
import importlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
IS_WINDOWS = sys.platform == "win32"
UINT64_MAX = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1
VECS_HEADER = struct.Struct("<QHHI11Q")
DEFAULT_REPO_ROOT = Path("D:/zvec-iaojnh")
DEFAULT_DATA_DIR = Path("D:/zvec_data")
DEFAULT_TRAIN_FILE = DEFAULT_DATA_DIR / "cohere_train_vector_1m.new.centaur.vecs"
DEFAULT_QUERY_FILE = DEFAULT_DATA_DIR / "cohere_test_vector_1000.new.txt"
DEFAULT_GROUND_TRUTH_FILE = DEFAULT_DATA_DIR / "ground_truth_d768_k100.txt"
DEFAULT_BUILD_DIR = DEFAULT_REPO_ROOT / "build"
DEFAULT_RESULTS_ROOT = DEFAULT_REPO_ROOT / "diskann_bench_windows"


def write_console(
    message: str = "",
    *,
    end: str = "\n",
    flush: bool = False,
    error: bool = False,
) -> None:
    """Write benchmark progress without routing it through Python logging."""

    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}{end}")
    if flush:
        stream.flush()


@dataclass
class ProcessMetrics:
    wall_seconds: float = 0.0
    peak_rss_mib: float | None = None
    read_operations: int | None = None
    read_megabytes: float | None = None
    read_iops: float | None = None
    read_mb_per_second: float | None = None


@dataclass(frozen=True)
class ProcessSample:
    current_rss_bytes: int
    peak_rss_bytes: int
    read_operations: int | None
    read_bytes: int | None


@dataclass
class BuildResult:
    precision: str
    index_path: str
    index_size_gib: float | None
    train_seconds: float | None
    build_seconds: float | None
    dump_seconds: float | None
    wall_seconds: float | None
    peak_rss_mib: float | None
    log_path: str


@dataclass
class RecallResult:
    precision: str
    list_size: int
    recall_at_1_pct: float
    recall_at_10_pct: float
    recall_at_50_pct: float
    query_count: int
    wall_seconds: float
    peak_rss_mib: float
    log_path: str


@dataclass
class SearchResult:
    precision: str
    list_size: int
    threads: int
    recall_at_1_pct: float
    recall_at_10_pct: float
    recall_at_50_pct: float
    qps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    query_count: int
    duration_ms: int
    peak_rss_mib: float
    recall_peak_rss_mib: float
    process_read_iops: float | None
    process_read_mb_per_second: float | None
    reads_per_query: float | None
    recall_log_path: str
    bench_log_path: str


@dataclass(frozen=True)
class VecsLayout:
    num_vecs: int
    dimension: int
    data_offset: int
    dense_offset: int
    key_offset: int


class IO_COUNTERS(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_sample_from_counters(
    memory: PROCESS_MEMORY_COUNTERS_EX, io: IO_COUNTERS | None
) -> ProcessSample:
    return ProcessSample(
        current_rss_bytes=int(memory.WorkingSetSize),
        peak_rss_bytes=int(memory.PeakWorkingSetSize),
        read_operations=int(io.ReadOperationCount) if io is not None else None,
        read_bytes=int(io.ReadTransferCount) if io is not None else None,
    )


def discover_repo_root() -> Path:
    """Find the repository even if this script was copied to another level."""

    script_path = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), script_path.parent, *script_path.parents]
    visited: set[Path] = set()
    for candidate in candidates:
        if candidate in visited:
            continue
        visited.add(candidate)
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    return Path.cwd().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Windows DiskANN build/recall/QPS matrix using the native "
            "zvec benchmark tools."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument(
        "--ground-truth-file", type=Path, default=DEFAULT_GROUND_TRUTH_FILE
    )
    parser.add_argument(
        "--ground-truth-mode",
        choices=("external", "generate", "internal"),
        default="generate",
        help=(
            "'external' requires --ground-truth-file; 'generate' creates an "
            "exact neighbors file with blocked NumPy matrix multiplication; "
            "'internal' uses recall_original's slower built-in linear scan."
        ),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Dense vector dimension used by the train/query dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Cohere 1M",
        help="Dataset label written to metadata and the Markdown report.",
    )
    parser.add_argument(
        "--ground-truth-k",
        type=int,
        default=100,
        help="Exact neighbors generated for each query.",
    )
    parser.add_argument(
        "--ground-truth-block-size",
        type=int,
        default=8192,
        help="Database vectors processed per exact-search matrix block.",
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="CMake build directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Result directory; defaults to "
            "D:/zvec-iaojnh/diskann_bench_windows/<timestamp>_<git-sha>."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Index directory; defaults to <output-dir>/indexes.",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("fp32", "fp16"),
        default=("fp32", "fp16"),
    )
    parser.add_argument("--list-sizes", nargs="+", type=int, default=(100, 300, 500))
    parser.add_argument("--thread-counts", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--build-threads", type=int, default=8)
    parser.add_argument(
        "--disable-id-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the supplied build YAML by disabling the builder ID map.",
    )
    parser.add_argument("--recall-threads", type=int, default=16)
    parser.add_argument("--bench-seconds", type=int, default=30)
    parser.add_argument("--bench-iterations", type=int, default=10_000_000)
    parser.add_argument("--cache-nodes", type=int, default=10_000)
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--builder-list-size", type=int, default=50)
    parser.add_argument("--pq-chunks", type=int, default=384)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=200_000,
        help="Maximum vectors sampled for DiskANN PQ training.",
    )
    parser.add_argument("--memory-limit", type=float, default=100.0)
    parser.add_argument("--top-k", default="1,10,50")
    parser.add_argument(
        "--server-label",
        default="",
        help="Optional server model written to the report, for example ecs.g9i.4xlarge.",
    )
    parser.add_argument(
        "--parallel-builds",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel jobs used when building benchmark tools.",
    )
    parser.add_argument(
        "--skip-tool-build",
        action="store_true",
        help="Require existing benchmark executables instead of building them.",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Reuse indexes from --index-dir; every requested index must exist.",
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help=(
            "Reuse recall_results.csv from the same --output-dir. Dataset, "
            "ground truth, index files, Git revision, and search parameters "
            "must match exactly."
        ),
    )
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Overwrite indexes that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create configs and print commands without executing them.",
    )
    return parser.parse_args(argv)


def require_positive(values: Iterable[int], label: str) -> None:
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"{label} must contain positive integers: {invalid}")


def resolved(path: Path, base: Path | None = None) -> Path:
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.expanduser().resolve()


def git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_branch(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "detached"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def count_query_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        count = sum(1 for line in stream if line.strip())
    if count == 0:
        raise ValueError(f"Query file contains no non-empty rows: {path}")
    return count


def append_msys2_to_path(env: dict[str, str]) -> None:
    """Append MSYS2 paths so Snowball can find make, Perl, and GCC.

    They are appended, not prepended, to avoid shadowing MSVC's link.exe.
    """

    candidates = (
        Path(r"C:\msys64\ucrt64\bin"),
        Path(r"C:\msys64\usr\bin"),
    )
    current_parts = env.get("PATH", "").split(os.pathsep)
    normalized = {os.path.normcase(part) for part in current_parts if part}
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and os.path.normcase(candidate_str) not in normalized:
            current_parts.append(candidate_str)
    env["PATH"] = os.pathsep.join(current_parts)


def executable(name: str, repo_root: Path, env: dict[str, str]) -> str:
    found = shutil.which(name, path=env.get("PATH"))
    if found:
        return found
    suffix = ".exe" if IS_WINDOWS else ""
    candidate = repo_root / ".venv" / "Scripts" / f"{name}{suffix}"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"Cannot find {name}. Activate .venv and install the build dependencies."
    )


def command_text(command: Sequence[str | Path]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_simple(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    write_console(f"\n> {command_text(command)}")
    if dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def check_x64_msvc(env: dict[str, str]) -> None:
    try:
        result = subprocess.run(
            ["cl"],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "MSVC cl.exe was not found. Run this script from "
            "'x64 Native Tools Command Prompt for VS 2022'."
        ) from exc
    output = f"{result.stdout}\n{result.stderr}"
    if not re.search(r"\bfor x64\b", output, re.IGNORECASE):
        raise RuntimeError(
            "The active MSVC compiler is not x64. Close this terminal and use "
            "'x64 Native Tools Command Prompt for VS 2022'."
        )


def find_tool(build_dir: Path, name: str) -> Path | None:
    filename = f"{name}.exe" if IS_WINDOWS else name
    preferred = (
        build_dir / "bin" / filename,
        build_dir / "bin" / "Release" / filename,
        build_dir / "Release" / filename,
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    if build_dir.is_dir():
        for candidate in build_dir.rglob(filename):
            if "CMakeFiles" not in candidate.parts and candidate.is_file():
                return candidate.resolve()
    return None


def ensure_tools(
    repo_root: Path,
    build_dir: Path,
    env: dict[str, str],
    *,
    skip_build: bool,
    parallel_builds: int,
    dry_run: bool,
) -> dict[str, Path]:
    names = ("local_builder", "recall_original", "bench_original")
    tools = {name: find_tool(build_dir, name) for name in names}
    if skip_build:
        missing = [name for name, path in tools.items() if path is None]
        if missing:
            raise FileNotFoundError(
                f"Missing benchmark tools in {build_dir}: {', '.join(missing)}"
            )
        return {name: path for name, path in tools.items() if path is not None}

    if IS_WINDOWS and not dry_run:
        check_x64_msvc(env)
    cmake = executable("cmake", repo_root, env)
    configure_command: list[str | Path] = [
        cmake,
        "-S",
        repo_root,
        "-B",
        build_dir,
    ]
    if not (build_dir / "CMakeCache.txt").exists():
        configure_command.extend(["-G", "Ninja"])
    configure_command.extend(
        [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TOOLS=ON",
            "-DBUILD_PYTHON_BINDINGS=OFF",
            "-DBUILD_C_BINDINGS=OFF",
            "-DBUILD_ZVEC_SHARED=OFF",
            "-DBUILD_ZVEC_CORE_SHARED=OFF",
            "-DBUILD_ZVEC_AILEGO_SHARED=OFF",
        ]
    )
    run_simple(configure_command, cwd=repo_root, env=env, dry_run=dry_run)
    run_simple(
        [
            cmake,
            "--build",
            build_dir,
            "--target",
            *names,
            "--config",
            "Release",
            "--parallel",
            str(parallel_builds),
        ],
        cwd=repo_root,
        env=env,
        dry_run=dry_run,
    )
    if dry_run:
        return {name: (build_dir / "bin" / f"{name}.exe").resolve() for name in names}
    tools = {name: find_tool(build_dir, name) for name in names}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Build completed but tools were not found: {', '.join(missing)}"
        )
    return {name: path for name, path in tools.items() if path is not None}


def yaml_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace("'", "''")
    return f"'{value}'"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def read_vecs_layout(train_file: Path, dimension: int) -> VecsLayout:
    """Read the offsets needed for dense FP32 vectors and uint64 keys."""

    with train_file.open("rb") as stream:
        raw_header = stream.read(VECS_HEADER.size)
    if len(raw_header) != VECS_HEADER.size:
        raise ValueError(f"Training file is too small for a VecsHeader: {train_file}")

    (
        num_vecs,
        _meta_size_v1,
        version,
        meta_size,
        bitmap,
        key_offset,
        key_size,
        dense_offset,
        dense_size,
        _sparse_offset,
        _sparse_size,
        _partition_offset,
        _partition_size,
        _taglist_offset,
        _taglist_size,
    ) = VECS_HEADER.unpack(raw_header)

    if version != 1:
        raise ValueError(
            "Automatic ground-truth generation currently requires a version "
            f"1 .vecs file; found version {version}. Supply "
            "--ground-truth-file or use --ground-truth-mode internal."
        )
    if num_vecs == 0:
        raise ValueError(f"Training file contains no vectors: {train_file}")
    if not bitmap & (1 << 1) or dense_offset == UINT64_MAX:
        raise ValueError("Training file does not contain dense vectors.")
    if not bitmap & (1 << 0) or key_offset == UINT64_MAX:
        raise ValueError("Training file does not contain uint64 vector keys.")

    dense_row_bytes, remainder = divmod(dense_size, num_vecs)
    if remainder or dense_row_bytes % 4:
        raise ValueError("Dense vector section is not a contiguous FP32 matrix.")
    stored_dimension = dense_row_bytes // 4
    if stored_dimension != dimension:
        raise ValueError(
            f"--dimension is {dimension}, but the training file stores "
            f"{stored_dimension} FP32 values per vector."
        )
    expected_key_size = num_vecs * 8
    if key_size < expected_key_size:
        raise ValueError(
            f"Key section is too small: expected at least "
            f"{expected_key_size} bytes, found {key_size}."
        )

    data_offset = VECS_HEADER.size + meta_size
    file_size = train_file.stat().st_size
    dense_end = data_offset + dense_offset + dense_size
    key_end = data_offset + key_offset + expected_key_size
    if max(dense_end, key_end) > file_size:
        raise ValueError("Training file header points beyond the end of the file.")

    return VecsLayout(
        num_vecs=num_vecs,
        dimension=stored_dimension,
        data_offset=data_offset,
        dense_offset=dense_offset,
        key_offset=key_offset,
    )


def load_query_matrix(
    query_file: Path, dimension: int, np: Any
) -> tuple[list[int], Any]:
    """Load the dense query field using recall_original's text convention."""

    query_ids: list[int] = []
    query_vectors: list[Any] = []
    with query_file.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(";")
            if len(fields) < 2:
                raise ValueError(
                    f"{query_file}:{line_number}: expected 'query_id;dense vector'."
                )
            try:
                query_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"{query_file}:{line_number}: invalid query id {fields[0]!r}."
                ) from exc
            vector = np.fromstring(fields[1], dtype=np.float32, sep=" ")
            if vector.size != dimension:
                raise ValueError(
                    f"{query_file}:{line_number}: expected {dimension} "
                    f"dense values, found {vector.size}."
                )
            query_ids.append(query_id)
            query_vectors.append(vector)

    if not query_vectors:
        raise ValueError(f"Query file contains no vectors: {query_file}")
    return query_ids, np.stack(query_vectors)


def display_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def generate_external_ground_truth(  # noqa: PLR0915
    *,
    train_file: Path,
    query_file: Path,
    output_file: Path,
    dimension: int,
    neighbor_count: int,
    block_size: int,
    dry_run: bool,
) -> None:
    """Generate exact cosine neighbors with bounded-memory matrix blocks."""

    if output_file.is_file():
        write_console(f"\nReusing ground truth: {output_file}")
        return
    if dry_run:
        write_console(
            "\nWould generate exact ground truth with NumPy: "
            f"{output_file} (dimension={dimension}, k={neighbor_count}, "
            f"block_size={block_size})"
        )
        return

    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for automatic ground-truth generation. "
            "Activate the project .venv and run 'python -m pip install "
            "numpy', or supply --ground-truth-file."
        ) from exc

    layout = read_vecs_layout(train_file, dimension)
    if neighbor_count > layout.num_vecs:
        raise ValueError(
            f"--ground-truth-k ({neighbor_count}) exceeds the number of "
            f"training vectors ({layout.num_vecs})."
        )

    query_ids, queries = load_query_matrix(query_file, dimension, np)
    query_norms = np.sqrt(np.einsum("ij,ij->i", queries, queries, optimize=True))
    if np.any(~np.isfinite(query_norms)) or np.any(query_norms == 0):
        raise ValueError("Query file contains a zero or non-finite vector.")
    queries /= query_norms[:, np.newaxis]
    query_transpose = np.ascontiguousarray(queries.T)
    query_count = len(query_ids)
    query_columns = np.arange(query_count)[np.newaxis, :]

    vectors = np.memmap(
        train_file,
        mode="r",
        dtype="<f4",
        offset=layout.data_offset + layout.dense_offset,
        shape=(layout.num_vecs, layout.dimension),
    )
    keys = np.memmap(
        train_file,
        mode="r",
        dtype="<u8",
        offset=layout.data_offset + layout.key_offset,
        shape=(layout.num_vecs,),
    )

    best_scores = np.full((neighbor_count, query_count), -np.inf, dtype=np.float32)
    best_rows = np.full((neighbor_count, query_count), -1, dtype=np.int64)
    started = time.perf_counter()
    write_console(
        f"\nGenerating exact cosine ground truth: "
        f"{layout.num_vecs:,} vectors x {query_count:,} queries, "
        f"k={neighbor_count}"
    )

    for start in range(0, layout.num_vecs, block_size):
        stop = min(start + block_size, layout.num_vecs)
        block = np.array(vectors[start:stop], dtype=np.float32, order="C", copy=True)
        block_norms = np.sqrt(np.einsum("ij,ij->i", block, block, optimize=True))
        invalid = ~np.isfinite(block_norms)
        if np.any(invalid):
            first_bad = start + int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Training vector row {first_bad} contains non-finite values."
            )
        block_norms[block_norms == 0] = 1.0
        block /= block_norms[:, np.newaxis]

        scores = block @ query_transpose
        block_k = min(neighbor_count, stop - start)
        local_rows = np.argpartition(scores, scores.shape[0] - block_k, axis=0)[
            -block_k:, :
        ]
        block_scores = scores[local_rows, query_columns]
        block_rows = local_rows.astype(np.int64, copy=False) + start

        candidate_scores = np.concatenate((best_scores, block_scores), axis=0)
        candidate_rows = np.concatenate((best_rows, block_rows), axis=0)
        keep = np.argpartition(
            candidate_scores,
            candidate_scores.shape[0] - neighbor_count,
            axis=0,
        )[-neighbor_count:, :]
        best_scores = candidate_scores[keep, query_columns]
        best_rows = candidate_rows[keep, query_columns]

        elapsed = time.perf_counter() - started
        completed = stop / layout.num_vecs
        eta = elapsed * (1.0 - completed) / completed
        write_console(
            f"\rGround truth: {completed:6.2%} "
            f"({stop:,}/{layout.num_vecs:,})  "
            f"elapsed {display_duration(elapsed)}, "
            f"ETA {display_duration(eta)}",
            end="",
            flush=True,
        )

    order = np.argsort(best_scores, axis=0)[::-1, :]
    best_rows = best_rows[order, query_columns]
    neighbor_keys = keys[np.ascontiguousarray(best_rows.T)]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="\n") as stream:
        for query_id, neighbor_row in zip(query_ids, neighbor_keys, strict=True):
            neighbors = " ".join(str(int(key)) for key in neighbor_row)
            stream.write(f"{query_id};{neighbors}\n")
    temporary_file.replace(output_file)

    elapsed = time.perf_counter() - started
    write_console(
        f"\nGround truth complete in {display_duration(elapsed)}: {output_file}"
    )


def build_yaml(
    *,
    train_file: Path,
    index_path: Path,
    converter: str,
    build_threads: int,
    max_degree: int,
    builder_list_size: int,
    memory_limit: float,
    pq_chunks: int,
    max_train_samples: int,
    disable_id_map: bool,
) -> str:
    disable_id_map_yaml = "true" if disable_id_map else "false"
    return f"""BuilderCommon:
    BuilderClass: DiskAnnBuilder
    BuildFile: {yaml_path(train_file)}
    NeedTrain: true
    TrainFile: {yaml_path(train_file)}
    DumpPath: {yaml_path(index_path)}
    IndexPath: {yaml_path(index_path)}
    MetricName: Cosine
    ConverterName: {converter}
    DisableIdMap: {disable_id_map_yaml}
    ThreadCount: {build_threads}
    LogLevel: Info
BuilderParams:
    zvec.general.builder.thread_count: !!int {build_threads}
    zvec.diskann.builder.thread_count: !!int {build_threads}
    zvec.diskann.builder.max_degree: !!int {max_degree}
    zvec.diskann.builder.list_size: !!int {builder_list_size}
    zvec.diskann.builder.memory_limit: !!float {memory_limit}
    zvec.diskann.builder.max_pq_chunk_num: !!int {pq_chunks}
    zvec.diskann.builder.max_train_sample_count: !!int {max_train_samples}
"""


def search_yaml(
    *,
    index_path: Path,
    query_file: Path,
    ground_truth_file: Path | None,
    recall_log_dir: Path,
    top_k: str,
    recall_gt_count: int,
    recall_threads: int,
    bench_threads: int,
    bench_seconds: int,
    bench_iterations: int,
    cache_nodes: int,
    list_size: int,
) -> str:
    ground_truth = ""
    if ground_truth_file is not None:
        ground_truth = f"    GroundTruthFile: {yaml_path(ground_truth_file)}\n"
    return f"""SearcherCommon:
    SearcherClass: DiskAnnSearcher
    IndexPath: {yaml_path(index_path)}
    TopK: {top_k}
    QueryFile: {yaml_path(query_file)}
    QueryType: float
    QueryFirstSep: ";"
    QuerySecondSep: " "
{ground_truth}    RecallLogDir: {yaml_path(recall_log_dir)}
    RecallThreadCount: {recall_threads}
    RecallGTCount: {recall_gt_count}
    RecallScorePrecision: 1e-4
    BenchThreadCount: {bench_threads}
    BenchSecs: {bench_seconds}
    BenchIterCount: {bench_iterations}
    CompareById: true
    ContainerType: FileReadStorage
    LogLevel: Info
SearcherParams:
    zvec.diskann.searcher.cache_node_num: !!int {cache_nodes}
    zvec.diskann.searcher.list_size: !!int {list_size}
ContainerParams: {{}}
"""


def process_sample(process: subprocess.Popen[str]) -> ProcessSample | None:
    if not IS_WINDOWS:
        return None
    try:
        handle = ctypes.wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        memory = PROCESS_MEMORY_COUNTERS_EX()
        memory.cb = ctypes.sizeof(memory)
        io = IO_COUNTERS()
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        memory_ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(memory), ctypes.sizeof(memory)
        )
        if not memory_ok:
            return None
        io_ok = kernel32.GetProcessIoCounters(handle, ctypes.byref(io))
        return process_sample_from_counters(memory, io if io_ok else None)
    except (AttributeError, OSError, ValueError):
        return None


def run_logged(  # noqa: PLR0915
    command: Sequence[str | Path],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    monitor_query: bool = False,
) -> ProcessMetrics:
    write_console(f"\n> {command_text(command)}")
    write_console(f"  log: {log_path}")
    if dry_run:
        return ProcessMetrics()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    query_started = threading.Event()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                write_console(line, end="")
                log_file.write(line)
                log_file.flush()
                if "Load index done!" in line:
                    query_started.set()

        output_thread = threading.Thread(target=copy_output, daemon=True)
        output_thread.start()
        peak_rss_bytes = 0
        query_baseline: ProcessSample | None = None
        query_baseline_time: float | None = None
        final_sample: ProcessSample | None = None
        try:
            while process.poll() is None:
                sample = process_sample(process)
                if sample is not None:
                    final_sample = sample
                    peak_rss_bytes = max(peak_rss_bytes, sample.peak_rss_bytes)
                    if (
                        monitor_query
                        and query_started.is_set()
                        and query_baseline is None
                    ):
                        query_baseline = sample
                        query_baseline_time = time.perf_counter()
                time.sleep(0.1)
        except KeyboardInterrupt:
            process.terminate()
            raise
        finally:
            output_thread.join(timeout=10)

        last_sample = process_sample(process)
        if last_sample is not None:
            final_sample = last_sample
            peak_rss_bytes = max(peak_rss_bytes, last_sample.peak_rss_bytes)
        return_code = process.wait()
        wall_seconds = time.perf_counter() - started
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, [str(part) for part in command]
            )

    metrics = ProcessMetrics(wall_seconds=wall_seconds)
    if peak_rss_bytes:
        metrics.peak_rss_mib = peak_rss_bytes / MIB
    elif IS_WINDOWS:
        raise RuntimeError(
            f"Failed to collect PeakWorkingSetSize for {command_text(command)}"
        )
    if (
        monitor_query
        and query_baseline is not None
        and query_baseline.read_operations is not None
        and query_baseline.read_bytes is not None
        and query_baseline_time is not None
        and final_sample is not None
        and final_sample.read_operations is not None
        and final_sample.read_bytes is not None
    ):
        duration = max(time.perf_counter() - query_baseline_time, 0.001)
        read_operations = max(
            0, final_sample.read_operations - query_baseline.read_operations
        )
        read_bytes = max(0, final_sample.read_bytes - query_baseline.read_bytes)
        metrics.read_operations = read_operations
        metrics.read_megabytes = read_bytes / MIB
        metrics.read_iops = read_operations / duration
        metrics.read_mb_per_second = (read_bytes / MIB) / duration
    return metrics


def read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def unique_match(pattern: str, text: str, label: str) -> tuple[str, ...]:
    matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(matches)}")
    match = matches[0]
    return (match,) if isinstance(match, str) else tuple(match)


def reject_error_log(text: str, label: str) -> None:
    markers = (
        r"^\s*\[\s*(?:ERROR|FATAL)\b",
        r"Failed to (?:knn[_ ]search|create search context|load ground truth file)",
        r"Load (?:query|ground truth file) (?:error|failed)",
        r"Flow load failed",
        r"Search results is small than queries",
        r"prefilter failed",
        r"query tag list not equal",
        r"Can not recognize type",
        r"unsupported retrieval mode",
    )
    for marker in markers:
        if re.search(marker, text, re.MULTILINE | re.IGNORECASE):
            raise ValueError(f"{label} log contains an error matching {marker!r}")


def parse_build_result(
    precision: str,
    index_path: Path,
    log_path: Path,
    metrics: ProcessMetrics | None,
    require_timings: bool,
) -> BuildResult:
    text = read_log(log_path)
    size = index_path.stat().st_size / GIB if index_path.is_file() else None
    if require_timings:
        reject_error_log(text, f"{precision} build")
        train_ms = float(
            unique_match(
                r"^Train finished,\s*consume\s+([0-9]+(?:\.[0-9]+)?)ms\.?$",
                text,
                "train timing",
            )[0]
        )
        build_ms = float(
            unique_match(
                r"^Build finished,\s*consume\s+([0-9]+(?:\.[0-9]+)?)ms\.?$",
                text,
                "build timing",
            )[0]
        )
        dump_ms = float(
            unique_match(
                r"^Dump to \[.*\] finished,\s*consume\s+"
                r"([0-9]+(?:\.[0-9]+)?)ms\.?$",
                text,
                "dump timing",
            )[0]
        )
        if size is None or size <= 0:
            raise ValueError(f"Build did not create a non-empty index: {index_path}")
        if metrics is None or metrics.peak_rss_mib is None:
            raise ValueError(f"Missing Peak RSS for {precision} build")
    else:
        train_ms = build_ms = dump_ms = None
    return BuildResult(
        precision=precision,
        index_path=str(index_path),
        index_size_gib=size,
        train_seconds=train_ms / 1000 if train_ms is not None else None,
        build_seconds=build_ms / 1000 if build_ms is not None else None,
        dump_seconds=dump_ms / 1000 if dump_ms is not None else None,
        wall_seconds=metrics.wall_seconds if metrics is not None else None,
        peak_rss_mib=metrics.peak_rss_mib if metrics is not None else None,
        log_path=str(log_path),
    )


def parse_recall_result(
    *,
    precision: str,
    list_size: int,
    log_path: Path,
    metrics: ProcessMetrics,
    expected_query_count: int,
    external_ground_truth: bool,
) -> RecallResult:
    text = read_log(log_path)
    reject_error_log(text, f"{precision} list_size={list_size} recall")
    values = re.findall(
        r"^Recall@(\d+):\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if len(values) != 3 or {int(k) for k, _ in values} != {1, 10, 50}:
        raise ValueError(
            f"Recall log must contain exactly Recall@1/@10/@50: {log_path}"
        )
    recall = {int(k): float(value) for k, value in values}
    if any(not 0.0 <= value <= 100.0 for value in recall.values()):
        raise ValueError(f"Recall percentage is outside [0, 100]: {recall}")
    query_count = int(
        unique_match(r"^Process query:\s*(\d+)\s*$", text, "recall query count")[0]
    )
    if query_count != expected_query_count:
        raise ValueError(
            f"Recall processed {query_count} queries, expected {expected_query_count}"
        )
    if "Load index done!" not in text or "Recall done." not in text:
        raise ValueError("Recall did not report successful index load and completion")
    if external_ground_truth and (
        "Load external ground truth file[" not in text
        or "Internal ground truth file NOT used" not in text
    ):
        raise ValueError("Recall did not use the configured external ground truth")
    if metrics.peak_rss_mib is None:
        raise ValueError("Recall Peak RSS was not collected")
    return RecallResult(
        precision=precision,
        list_size=list_size,
        recall_at_1_pct=recall[1],
        recall_at_10_pct=recall[10],
        recall_at_50_pct=recall[50],
        query_count=query_count,
        wall_seconds=metrics.wall_seconds,
        peak_rss_mib=metrics.peak_rss_mib,
        log_path=str(log_path),
    )


def parse_search_result(
    *,
    precision: str,
    list_size: int,
    threads: int,
    recall: RecallResult,
    recall_log_path: Path,
    bench_log_path: Path,
    metrics: ProcessMetrics,
    bench_seconds: int,
) -> SearchResult:
    text = read_log(bench_log_path)
    reject_error_log(text, f"{precision} list_size={list_size} threads={threads} bench")
    process_values = unique_match(
        r"^Process query:\s*(\d+), total process time:\s*(\d+)ms, "
        r"duration:\s*(\d+)ms, max:\s*(\d+)ms, min:\s*(\d+)ms\s*$",
        text,
        "benchmark process summary",
    )
    query_count, total_ms, duration_ms, max_latency_ms, min_latency_ms = map(
        int, process_values
    )
    avg_latency_ms, qps = map(
        float,
        unique_match(
            r"^Avg latency:\s*([0-9]+(?:\.[0-9]+)?)ms\s+qps:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*$",
            text,
            "average latency/QPS summary",
        ),
    )
    percentile_matches = re.findall(
        r"^(25|50|75|90|95|99) Percentile:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*ms\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if len(percentile_matches) != 6:
        raise ValueError(
            f"Benchmark must report each percentile exactly once: {percentile_matches}"
        )
    percentiles = {
        int(percentile): float(value) for percentile, value in percentile_matches
    }
    if set(percentiles) != {25, 50, 75, 90, 95, 99}:
        raise ValueError(f"Benchmark percentile set is incomplete: {percentiles}")
    if list(percentiles.values()) != sorted(percentiles.values()):
        raise ValueError(f"Benchmark percentiles are not monotonic: {percentiles}")
    if query_count <= 0 or duration_ms < bench_seconds * 900 or qps <= 0:
        raise ValueError(
            f"Benchmark ended too early: queries={query_count}, duration_ms={duration_ms}"
        )
    calculated_qps = query_count * 1000.0 / duration_ms
    if abs(calculated_qps - qps) > max(0.2, qps * 0.05):
        raise ValueError(f"Reported QPS {qps} disagrees with {calculated_qps:.3f}")
    calculated_avg = total_ms / query_count
    if abs(calculated_avg - avg_latency_ms) > max(0.5, avg_latency_ms * 0.1):
        raise ValueError(
            f"Reported average latency {avg_latency_ms} disagrees with "
            f"{calculated_avg:.3f}"
        )
    if "Load index done!" not in text or "Bench done." not in text:
        raise ValueError(
            "Benchmark did not report successful index load and completion"
        )
    if metrics.peak_rss_mib is None:
        raise ValueError("Benchmark Peak RSS was not collected")
    read_operations = metrics.read_operations
    reads_per_query = None
    if read_operations is not None:
        reads_per_query = read_operations / query_count
    return SearchResult(
        precision=precision,
        list_size=list_size,
        threads=threads,
        recall_at_1_pct=recall.recall_at_1_pct,
        recall_at_10_pct=recall.recall_at_10_pct,
        recall_at_50_pct=recall.recall_at_50_pct,
        qps=qps,
        avg_latency_ms=avg_latency_ms,
        p50_latency_ms=percentiles[50],
        p95_latency_ms=percentiles[95],
        p99_latency_ms=percentiles[99],
        min_latency_ms=float(min_latency_ms),
        max_latency_ms=float(max_latency_ms),
        query_count=query_count,
        duration_ms=duration_ms,
        peak_rss_mib=metrics.peak_rss_mib,
        recall_peak_rss_mib=recall.peak_rss_mib,
        process_read_iops=metrics.read_iops,
        process_read_mb_per_second=metrics.read_mb_per_second,
        reads_per_query=reads_per_query,
        recall_log_path=str(recall_log_path),
        bench_log_path=str(bench_log_path),
    )


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_recall_results(path: Path) -> list[RecallResult]:
    if not path.is_file():
        raise FileNotFoundError(
            f"--skip-recall requires an existing result file: {path}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        RecallResult(
            precision=row["precision"],
            list_size=int(row["list_size"]),
            recall_at_1_pct=float(row["recall_at_1_pct"]),
            recall_at_10_pct=float(row["recall_at_10_pct"]),
            recall_at_50_pct=float(row["recall_at_50_pct"]),
            query_count=int(row["query_count"]),
            wall_seconds=float(row["wall_seconds"]),
            peak_rss_mib=float(row["peak_rss_mib"]),
            log_path=row["log_path"],
        )
        for row in rows
    ]


def file_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"--skip-recall requires existing metadata from this script: {path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must contain a JSON object: {path}")
    return value


def validate_reused_recall(
    previous: dict[str, Any],
    current: dict[str, Any],
    recalls: list[RecallResult],
) -> None:
    """Reject old Recall data unless its full provenance matches this run."""

    top_level_keys = (
        "git_sha",
        "git_dirty",
        "query_file",
        "query_count",
        "query_file_signature",
        "ground_truth_file",
        "ground_truth_file_signature",
        "index_files",
    )
    parameter_keys = (
        "precision",
        "list_sizes",
        "cache_nodes",
        "top_k",
        "ground_truth_mode",
        "dimension",
        "ground_truth_k",
    )
    mismatches = [
        key for key in top_level_keys if previous.get(key) != current.get(key)
    ]
    previous_parameters = previous.get("parameters", {})
    current_parameters = current.get("parameters", {})
    mismatches.extend(
        f"parameters.{key}"
        for key in parameter_keys
        if previous_parameters.get(key) != current_parameters.get(key)
    )
    if mismatches:
        raise ValueError(
            "--skip-recall cannot reuse results because provenance changed: "
            + ", ".join(mismatches)
        )
    if current.get("git_dirty") is not False:
        raise ValueError(
            "--skip-recall requires a clean Git worktree so code provenance "
            "can be verified."
        )

    expected_query_count = int(current["query_count"])
    for row in recalls:
        values = (
            row.recall_at_1_pct,
            row.recall_at_10_pct,
            row.recall_at_50_pct,
            row.wall_seconds,
            row.peak_rss_mib,
        )
        if row.query_count != expected_query_count:
            raise ValueError(
                f"Cached Recall processed {row.query_count} queries; "
                f"this run has {expected_query_count}."
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Cached Recall contains a non-finite value: {row}")
        if not all(0.0 <= value <= 100.0 for value in values[:3]):
            raise ValueError(f"Cached Recall percentage is outside [0, 100]: {row}")
        if row.wall_seconds <= 0 or row.peak_rss_mib <= 0:
            raise ValueError(f"Cached Recall timing/RSS must be positive: {row}")


def validate_result_contract(
    metadata: dict[str, Any],
    builds: list[BuildResult],
    recalls: list[RecallResult],
    searches: list[SearchResult],
    *,
    require_recalls: bool,
    require_searches: bool,
) -> None:
    parameters = metadata["parameters"]
    precisions = list(parameters["precision"])
    list_sizes = [int(value) for value in parameters["list_sizes"]]
    thread_counts = [int(value) for value in parameters["thread_counts"]]
    if {row.precision for row in builds} != set(precisions):
        raise ValueError("Build results do not cover every requested precision")
    expected_recall = {
        (precision, list_size) for precision in precisions for list_size in list_sizes
    }
    actual_recall = {(row.precision, row.list_size) for row in recalls}
    if require_recalls and actual_recall != expected_recall:
        raise ValueError(
            f"Recall result matrix is incomplete: expected={expected_recall}, "
            f"actual={actual_recall}"
        )
    expected_search = {
        (precision, list_size, threads)
        for precision in precisions
        for list_size in list_sizes
        for threads in thread_counts
    }
    actual_search = {(row.precision, row.list_size, row.threads) for row in searches}
    if require_searches and actual_search != expected_search:
        raise ValueError(
            f"Search result matrix is incomplete: expected={expected_search}, "
            f"actual={actual_search}"
        )
    expected_query_count = int(metadata["query_count"])
    for row in recalls:
        recall_values = (
            row.recall_at_1_pct,
            row.recall_at_10_pct,
            row.recall_at_50_pct,
        )
        if row.query_count != expected_query_count:
            raise ValueError(
                f"Recall query count must be {expected_query_count}: {row}"
            )
        if not all(math.isfinite(value) for value in recall_values):
            raise ValueError(f"Recall values must be finite: {row}")
        if not all(0.0 <= value <= 100.0 for value in recall_values):
            raise ValueError(f"Recall values must be percentages: {row}")
        if row.peak_rss_mib <= 0:
            raise ValueError(f"Recall Peak RSS must be positive: {row}")
    for row in searches:
        if row.qps <= 0 or row.peak_rss_mib <= 0:
            raise ValueError(f"QPS and Peak RSS must be positive: {row}")


def write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    builds: list[BuildResult],
    recalls: list[RecallResult],
    searches: list[SearchResult],
) -> None:
    write_text(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    build_rows = [asdict(row) for row in builds]
    recall_rows = [asdict(row) for row in recalls]
    search_rows = [asdict(row) for row in searches]
    write_csv(output_dir / "build_results.csv", build_rows)
    write_csv(output_dir / "recall_results.csv", recall_rows)
    write_csv(output_dir / "results.csv", search_rows)
    write_text(
        output_dir / "results.json",
        json.dumps(
            {"builds": build_rows, "recalls": recall_rows, "searches": search_rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )

    lines = [
        "# Zvec DiskANN Windows benchmark",
        "",
        "测试参数与 Cohere 1M Windows FP32/FP16 DiskANN 测试文档对齐。",
        "",
        "## Environment",
        "",
        f"- Time: {metadata.get('timestamp', '')}",
        f"- Git: `{metadata.get('git_sha', '')}` ({metadata.get('git_branch', '')})",
        f"- Git dirty: {metadata.get('git_dirty', '')}",
        f"- Server: {metadata.get('server_label', '')}",
        f"- OS: {metadata.get('platform', '')}",
        f"- CPU: {metadata.get('processor', '')}",
        f"- Logical CPUs: {metadata.get('logical_cpu_count', '')}",
        f"- Python: {metadata.get('python', '')}",
        f"- I/O backend: `{metadata.get('io_backend', '')}`",
        f"- I/O description: {metadata.get('io_backend_description', '')}",
        f"- I/O backend source: {metadata.get('io_backend_source', '')}",
        f"- Ground truth: {metadata.get('ground_truth_file', '')}",
        "",
        "## Test data",
        "",
        f"- Dataset: {metadata.get('dataset_name', '')}",
        f"- Training records: {metadata.get('train_record_count', '')}",
        f"- Dimension: {metadata.get('parameters', {}).get('dimension', '')}",
        f"- Queries: {metadata.get('query_count', '')}",
        f"- Metric: {metadata.get('metric', '')}",
        f"- Converters: {metadata.get('converters', '')}",
        "",
        "## Parameters",
        "",
        f"- Build type: {metadata.get('build_type', '')}",
        f"- Build threads: {metadata.get('parameters', {}).get('build_threads', '')}",
        f"- Disable ID map: {metadata.get('parameters', {}).get('disable_id_map', '')}",
        f"- Max degree: {metadata.get('parameters', {}).get('max_degree', '')}",
        f"- Builder list size: {metadata.get('parameters', {}).get('builder_list_size', '')}",
        f"- PQ chunks: {metadata.get('parameters', {}).get('pq_chunks', '')}",
        f"- Max train samples: {metadata.get('parameters', {}).get('max_train_samples', '')}",
        f"- Memory limit: {metadata.get('parameters', {}).get('memory_limit', '')}",
        f"- Cache nodes: {metadata.get('parameters', {}).get('cache_nodes', '')}",
        f"- Benchmark TopK: {metadata.get('parameters', {}).get('bench_top_k', '')}",
        "",
        "## Build",
        "",
        "| Precision | Index GiB | Train s | Build s | Dump s | Wall s | Peak RSS MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in builds:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.precision.upper(),
                    fmt(row.index_size_gib, 2),
                    fmt(row.train_seconds, 3),
                    fmt(row.build_seconds, 3),
                    fmt(row.dump_seconds, 3),
                    fmt(row.wall_seconds, 3),
                    fmt(row.peak_rss_mib, 1),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recall",
            "",
            "| Precision | List | R@1 % | R@10 % | R@50 % | Queries | Peak RSS MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in recalls:
        lines.append(
            f"| {row.precision.upper()} | {row.list_size} | "
            f"{fmt(row.recall_at_1_pct, 3)} | {fmt(row.recall_at_10_pct, 3)} | "
            f"{fmt(row.recall_at_50_pct, 3)} | {row.query_count} | "
            f"{fmt(row.peak_rss_mib, 1)} |"
        )
    lines.extend(
        [
            "",
            "## QPS and RSS",
            "",
            (
                "| Precision | List | Threads | R@1 % | R@10 % | R@50 % | "
                "QPS | Avg ms | P50 ms | P95 ms | P99 ms | Peak RSS MiB | "
                "Read IOPS | Read MiB/s | Reads/query |"
            ),
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            ),
        ]
    )
    for row in searches:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.precision.upper(),
                    str(row.list_size),
                    str(row.threads),
                    fmt(row.recall_at_1_pct, 3),
                    fmt(row.recall_at_10_pct, 3),
                    fmt(row.recall_at_50_pct, 3),
                    fmt(row.qps, 1),
                    fmt(row.avg_latency_ms, 1),
                    fmt(row.p50_latency_ms, 1),
                    fmt(row.p95_latency_ms, 1),
                    fmt(row.p99_latency_ms, 1),
                    fmt(row.peak_rss_mib, 1),
                    fmt(row.process_read_iops, 1),
                    fmt(row.process_read_mb_per_second, 1),
                    fmt(row.reads_per_query, 1),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "> Read IOPS / MiB/s come from the benchmark process's "
                "Windows I/O counters after `Load index done!`. With the "
                "DiskANN reader's `FILE_FLAG_NO_BUFFERING`, they describe the "
                "query read workload, but they are not device-wide hardware "
                "counters."
            ),
            (
                "> Peak RSS is the Windows `PeakWorkingSetSize` high-water mark "
                "for the complete child process, including index load and cache preload."
            ),
            "",
            (
                "> QPS uses the largest requested TopK. Recall values are percentages "
                "reported directly by `recall_original`."
            ),
            (
                "> Native tools are `local_builder`, `recall_original`, and "
                "`bench_original`."
            ),
            "",
        ]
    )
    write_text(output_dir / "summary.md", "\n".join(lines))
    write_document_matrix(output_dir, metadata, builds, recalls, searches)


def write_document_matrix(
    output_dir: Path,
    metadata: dict[str, Any],
    builds: list[BuildResult],
    recalls: list[RecallResult],
    searches: list[SearchResult],
) -> None:
    parameters = metadata.get("parameters", {})
    columns = [
        (precision, int(list_size))
        for precision in parameters.get("precision", [])
        for list_size in parameters.get("list_sizes", [])
    ]
    if not columns:
        return
    build_by_precision = {row.precision: row for row in builds}
    recall_by_key = {(row.precision, row.list_size): row for row in recalls}
    search_by_key = {
        (row.precision, row.list_size, row.threads): row for row in searches
    }

    def build_value(precision: str, field: str, digits: int) -> str:
        row = build_by_precision.get(precision)
        return fmt(getattr(row, field, None), digits) if row else ""

    def recall_value(precision: str, list_size: int, field: str) -> str:
        row = recall_by_key.get((precision, list_size))
        return fmt(getattr(row, field, None), 3) if row else ""

    def search_value(
        precision: str, list_size: int, threads: int, field: str, digits: int
    ) -> str:
        row = search_by_key.get((precision, list_size, threads))
        return fmt(getattr(row, field, None), digits) if row else ""

    matrix: list[tuple[str, list[str]]] = [
        (
            "Index GiB",
            [build_value(precision, "index_size_gib", 2) for precision, _ in columns],
        ),
        (
            "Train s",
            [build_value(precision, "train_seconds", 3) for precision, _ in columns],
        ),
        (
            "Build s",
            [build_value(precision, "build_seconds", 3) for precision, _ in columns],
        ),
        (
            "Dump s",
            [build_value(precision, "dump_seconds", 3) for precision, _ in columns],
        ),
        (
            "Build Peak RSS MiB",
            [build_value(precision, "peak_rss_mib", 1) for precision, _ in columns],
        ),
    ]
    for top_k, field in (
        (1, "recall_at_1_pct"),
        (10, "recall_at_10_pct"),
        (50, "recall_at_50_pct"),
    ):
        matrix.append(
            (
                f"Recall@{top_k} %",
                [
                    recall_value(precision, list_size, field)
                    for precision, list_size in columns
                ],
            )
        )
    for threads in parameters.get("thread_counts", []):
        thread_count = int(threads)
        matrix.extend(
            [
                (
                    f"QPS ({thread_count} thread)",
                    [
                        search_value(precision, list_size, thread_count, "qps", 1)
                        for precision, list_size in columns
                    ],
                ),
                (
                    f"Peak RSS MiB ({thread_count} thread)",
                    [
                        search_value(
                            precision, list_size, thread_count, "peak_rss_mib", 1
                        )
                        for precision, list_size in columns
                    ],
                ),
            ]
        )

    column_names = [
        f"{precision.upper()} L{list_size}" for precision, list_size in columns
    ]
    csv_rows = [
        {"metric": metric, **dict(zip(column_names, values, strict=True))}
        for metric, values in matrix
    ]
    write_csv(output_dir / "document_matrix.csv", csv_rows)
    markdown = [
        "# DiskANN Windows document-ready results",
        "",
        "| Metric | " + " | ".join(column_names) + " |",
        "| --- | " + " | ".join("---:" for _ in column_names) + " |",
    ]
    markdown.extend(
        f"| {metric} | " + " | ".join(values) + " |" for metric, values in matrix
    )
    markdown.extend(
        [
            "",
            "> RSS is PeakWorkingSetSize for the complete process.",
            f"> QPS is measured at TopK={parameters.get('bench_top_k', '')}.",
            "",
        ]
    )
    write_text(output_dir / "document_results.md", "\n".join(markdown))


def get_zvec_backend() -> tuple[str, str]:
    try:
        zvec = importlib.import_module("zvec")

        return str(zvec.io_backend_type()), str(zvec.io_backend_description())
    except Exception as exc:  # pragma: no cover - environment dependent
        return "unavailable", f"{type(exc).__name__}: {exc}"


def update_native_io_backend(metadata: dict[str, Any], log_path: Path) -> bool:
    match = re.search(
        r"DiskAnn: I/O backend '([^']+)'\s*[—-]\s*(.+?)\s*$",
        read_log(log_path),
        re.MULTILINE,
    )
    if match:
        metadata["io_backend"] = match.group(1)
        metadata["io_backend_description"] = match.group(2)
        metadata["io_backend_source"] = str(log_path)
        return True
    return False


def require_windows_overlapped_backend(
    metadata: dict[str, Any], log_path: Path
) -> None:
    found = update_native_io_backend(metadata, log_path)
    if IS_WINDOWS and (not found or metadata.get("io_backend") != "windows_overlapped"):
        raise ValueError(
            "Native benchmark did not confirm the windows_overlapped I/O "
            f"backend; check that the intended Release tools were built: {log_path}"
        )


def get_disk_metadata() -> Any:
    if not IS_WINDOWS:
        return []
    powershell = shutil.which("powershell.exe")
    if not powershell:
        return []
    command = (
        "Get-CimInstance Win32_DiskDrive | "
        "Select-Object Model,InterfaceType,MediaType,Size | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
        output = result.stdout.strip()
        return json.loads(output) if output else []
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []


def main() -> int:  # noqa: PLR0915
    args = parse_args()
    repo_root = resolved(args.repo_root)
    train_file = resolved(args.train_file)
    query_file = resolved(args.query_file)
    ground_truth_file = (
        resolved(args.ground_truth_file) if args.ground_truth_file else None
    )
    build_dir = resolved(args.build_dir, repo_root)

    require_positive(args.list_sizes, "list sizes")
    require_positive(args.thread_counts, "thread counts")
    require_positive(
        (
            args.build_threads,
            args.recall_threads,
            args.bench_seconds,
            args.bench_iterations,
            args.parallel_builds,
            args.dimension,
            args.ground_truth_k,
            args.ground_truth_block_size,
            args.max_degree,
            args.builder_list_size,
            args.pq_chunks,
            args.max_train_samples,
        ),
        "benchmark parameters",
    )
    if not math.isfinite(args.memory_limit) or args.memory_limit <= 0:
        raise ValueError("--memory-limit must be finite and positive")
    if args.cache_nodes < 0:
        raise ValueError("--cache-nodes must be non-negative")
    if args.max_train_samples > UINT32_MAX:
        raise ValueError(f"--max-train-samples must be at most {UINT32_MAX}")
    try:
        requested_top_k = [int(value) for value in args.top_k.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--top-k must be a comma-separated integer list: {args.top_k!r}"
        ) from exc
    require_positive(requested_top_k, "top-k values")
    if sorted(set(requested_top_k)) != [1, 10, 50]:
        raise ValueError("--top-k must contain exactly 1,10,50 for this report")
    if args.ground_truth_k < max(requested_top_k):
        raise ValueError(
            f"--ground-truth-k ({args.ground_truth_k}) must be at least the "
            f"largest --top-k value ({max(requested_top_k)})."
        )
    if (
        not args.skip_recall
        and args.ground_truth_mode == "external"
        and ground_truth_file is None
    ):
        raise ValueError(
            "--ground-truth-file is required unless --ground-truth-mode is "
            "generate or internal"
        )
    if args.ground_truth_mode == "internal":
        ground_truth_file = None
    if args.skip_recall and args.rebuild_index:
        raise ValueError("--skip-recall cannot be combined with --rebuild-index")
    if not args.dry_run and not IS_WINDOWS:
        raise RuntimeError("This benchmark runner must be executed on Windows.")
    for label, path in (
        ("repository", repo_root),
        ("train file", train_file),
        ("query file", query_file),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if (
        ground_truth_file is not None
        and args.ground_truth_mode == "external"
        and not ground_truth_file.is_file()
    ):
        raise FileNotFoundError(
            f"ground-truth file does not exist: {ground_truth_file}"
        )

    sha = git_sha(repo_root)
    branch = git_branch(repo_root)
    dirty = git_dirty(repo_root)
    stamp = dt.datetime.now(tz=dt.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolved(args.output_dir, repo_root)
        if args.output_dir
        else (DEFAULT_RESULTS_ROOT / f"{stamp}_{sha}").resolve()
    )
    index_dir = (
        resolved(args.index_dir, repo_root)
        if args.index_dir
        else output_dir / "indexes"
    )
    if args.dry_run and output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(
            "--dry-run refuses to overwrite a non-empty --output-dir; choose "
            "a new directory."
        )
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    recall_detail_dir = output_dir / "recall"
    for directory in (
        output_dir,
        index_dir,
        config_dir,
        log_dir,
        recall_detail_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_recall and args.ground_truth_mode == "generate":
        if ground_truth_file is None:
            ground_truth_file = (
                output_dir
                / f"ground_truth_d{args.dimension}_k{args.ground_truth_k}.txt"
            ).resolve()
        generate_external_ground_truth(
            train_file=train_file,
            query_file=query_file,
            output_file=ground_truth_file,
            dimension=args.dimension,
            neighbor_count=args.ground_truth_k,
            block_size=args.ground_truth_block_size,
            dry_run=args.dry_run,
        )

    expected_query_count = count_query_rows(query_file)
    train_layout = (
        None if args.dry_run else read_vecs_layout(train_file, args.dimension)
    )
    bench_top_k = max(requested_top_k)
    env = os.environ.copy()
    append_msys2_to_path(env)
    io_backend, io_description = get_zvec_backend()
    indexes = {
        precision: (index_dir / f"diskann_{precision}.index").resolve()
        for precision in args.precision
    }
    metadata: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": dirty,
        "server_label": args.server_label,
        "dataset_name": args.dataset_name,
        "build_type": "Release",
        "metric": "Cosine",
        "converters": {
            "fp32": "CosineFp32Converter",
            "fp16": "CosineFp16Converter",
        },
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "io_backend": io_backend,
        "io_backend_description": io_description,
        "io_backend_source": "active Python environment",
        "disks": get_disk_metadata(),
        "train_file": str(train_file),
        "train_file_gib": (
            train_file.stat().st_size / GIB if train_file.is_file() else None
        ),
        "train_record_count": train_layout.num_vecs if train_layout else None,
        "query_file": str(query_file),
        "query_count": expected_query_count,
        "query_file_signature": file_signature(query_file),
        "ground_truth_file": (str(ground_truth_file) if ground_truth_file else None),
        "ground_truth_file_signature": file_signature(ground_truth_file),
        "index_files": {
            precision: file_signature(path) for precision, path in indexes.items()
        },
        "parameters": {
            "precision": list(args.precision),
            "list_sizes": list(args.list_sizes),
            "thread_counts": list(args.thread_counts),
            "build_threads": args.build_threads,
            "disable_id_map": args.disable_id_map,
            "recall_threads": args.recall_threads,
            "bench_seconds": args.bench_seconds,
            "bench_iterations": args.bench_iterations,
            "cache_nodes": args.cache_nodes,
            "max_degree": args.max_degree,
            "builder_list_size": args.builder_list_size,
            "pq_chunks": args.pq_chunks,
            "max_train_samples": args.max_train_samples,
            "memory_limit": args.memory_limit,
            "top_k": args.top_k,
            "bench_top_k": bench_top_k,
            "ground_truth_mode": args.ground_truth_mode,
            "dimension": args.dimension,
            "ground_truth_k": args.ground_truth_k,
            "ground_truth_block_size": args.ground_truth_block_size,
        },
    }
    recall_rows: list[RecallResult] = []
    if args.skip_recall and not args.skip_bench and not args.dry_run:
        previous_metadata = load_metadata(output_dir / "metadata.json")
        recall_rows = load_recall_results(output_dir / "recall_results.csv")
        validate_reused_recall(previous_metadata, metadata, recall_rows)
    write_outputs(output_dir, metadata, [], recall_rows, [])

    tools = ensure_tools(
        repo_root,
        build_dir,
        env,
        skip_build=args.skip_tool_build,
        parallel_builds=args.parallel_builds,
        dry_run=args.dry_run,
    )
    write_console(f"\nResults: {output_dir}")
    write_console(f"I/O backend: {io_backend}")

    converters = {
        "fp32": "CosineFp32Converter",
        "fp16": "CosineFp16Converter",
    }
    build_results: list[BuildResult] = []
    for precision in args.precision:
        index_path = indexes[precision]
        config_path = config_dir / f"build_{precision}.yaml"
        write_text(
            config_path,
            build_yaml(
                train_file=train_file,
                index_path=index_path,
                converter=converters[precision],
                build_threads=args.build_threads,
                max_degree=args.max_degree,
                builder_list_size=args.builder_list_size,
                memory_limit=args.memory_limit,
                pq_chunks=args.pq_chunks,
                max_train_samples=args.max_train_samples,
                disable_id_map=args.disable_id_map,
            ),
        )
        build_log = log_dir / f"build_{precision}.log"
        build_metrics: ProcessMetrics | None = None
        should_build = not args.skip_index_build and (
            args.rebuild_index or not index_path.exists()
        )
        if should_build:
            if args.rebuild_index and index_path.exists() and not args.dry_run:
                if not index_path.is_file():
                    raise RuntimeError(
                        f"Refusing to replace a non-file index path: {index_path}"
                    )
                index_path.unlink()
            build_metrics = run_logged(
                [tools["local_builder"], config_path],
                log_path=build_log,
                cwd=output_dir,
                env=env,
                dry_run=args.dry_run,
            )
        elif index_path.exists():
            write_console(f"\nReusing existing {precision.upper()} index: {index_path}")
        elif not args.dry_run:
            raise FileNotFoundError(
                f"{precision.upper()} index is missing: {index_path}"
            )
        build_results.append(
            parse_build_result(
                precision,
                index_path,
                build_log,
                build_metrics,
                require_timings=should_build and not args.dry_run,
            )
        )
        write_outputs(output_dir, metadata, build_results, recall_rows, [])

    metadata["index_files"] = {
        precision: file_signature(path) for precision, path in indexes.items()
    }
    recall_results: dict[tuple[str, int], RecallResult] = {
        (row.precision, row.list_size): row for row in recall_rows
    }
    recall_logs: dict[tuple[str, int], Path] = {
        (row.precision, row.list_size): Path(row.log_path) for row in recall_rows
    }
    if not args.skip_recall:
        recall_rows.clear()
        recall_results.clear()
        recall_logs.clear()
        for precision in args.precision:
            for list_size in args.list_sizes:
                recall_log = log_dir / f"recall_{precision}_l{list_size}.log"
                recall_logs[(precision, list_size)] = recall_log
                config_path = config_dir / f"recall_{precision}_l{list_size}.yaml"
                write_text(
                    config_path,
                    search_yaml(
                        index_path=indexes[precision],
                        query_file=query_file,
                        ground_truth_file=ground_truth_file,
                        recall_log_dir=(
                            recall_detail_dir / f"{precision}_l{list_size}"
                        ),
                        top_k=args.top_k,
                        recall_gt_count=args.ground_truth_k,
                        recall_threads=args.recall_threads,
                        bench_threads=max(args.thread_counts),
                        bench_seconds=args.bench_seconds,
                        bench_iterations=args.bench_iterations,
                        cache_nodes=args.cache_nodes,
                        list_size=list_size,
                    ),
                )
                recall_metrics = run_logged(
                    [tools["recall_original"], config_path],
                    log_path=recall_log,
                    cwd=output_dir,
                    env=env,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    result = parse_recall_result(
                        precision=precision,
                        list_size=list_size,
                        log_path=recall_log,
                        metrics=recall_metrics,
                        expected_query_count=expected_query_count,
                        external_ground_truth=ground_truth_file is not None,
                    )
                    recall_rows.append(result)
                    recall_results[(precision, list_size)] = result
                    require_windows_overlapped_backend(metadata, recall_log)
                    write_outputs(output_dir, metadata, build_results, recall_rows, [])

    search_results: list[SearchResult] = []
    if not args.skip_bench:
        for precision in args.precision:
            for list_size in args.list_sizes:
                for threads in args.thread_counts:
                    bench_log = (
                        log_dir / f"bench_{precision}_l{list_size}_t{threads}.log"
                    )
                    config_path = (
                        config_dir / f"bench_{precision}_l{list_size}_t{threads}.yaml"
                    )
                    write_text(
                        config_path,
                        search_yaml(
                            index_path=indexes[precision],
                            query_file=query_file,
                            ground_truth_file=ground_truth_file,
                            recall_log_dir=(
                                recall_detail_dir / f"{precision}_l{list_size}"
                            ),
                            top_k=str(bench_top_k),
                            recall_gt_count=args.ground_truth_k,
                            recall_threads=args.recall_threads,
                            bench_threads=threads,
                            bench_seconds=args.bench_seconds,
                            bench_iterations=args.bench_iterations,
                            cache_nodes=args.cache_nodes,
                            list_size=list_size,
                        ),
                    )
                    metrics = run_logged(
                        [tools["bench_original"], config_path],
                        log_path=bench_log,
                        cwd=output_dir,
                        env=env,
                        dry_run=args.dry_run,
                        monitor_query=True,
                    )
                    if not args.dry_run:
                        require_windows_overlapped_backend(metadata, bench_log)
                        search_results.append(
                            parse_search_result(
                                precision=precision,
                                list_size=list_size,
                                threads=threads,
                                recall=recall_results[(precision, list_size)],
                                recall_log_path=recall_logs[(precision, list_size)],
                                bench_log_path=bench_log,
                                metrics=metrics,
                                bench_seconds=args.bench_seconds,
                            )
                        )
                        write_outputs(
                            output_dir,
                            metadata,
                            build_results,
                            recall_rows,
                            search_results,
                        )

    if not args.dry_run:
        validate_result_contract(
            metadata,
            build_results,
            recall_rows,
            search_results,
            require_recalls=not args.skip_recall or not args.skip_bench,
            require_searches=not args.skip_bench,
        )
    write_outputs(output_dir, metadata, build_results, recall_rows, search_results)
    write_console("\nBenchmark complete.")
    write_console(f"Markdown summary: {output_dir / 'summary.md'}")
    write_console(f"Document table:   {output_dir / 'document_results.md'}")
    if (output_dir / "results.csv").is_file():
        write_console(f"CSV results:      {output_dir / 'results.csv'}")
    else:
        write_console("CSV results:      not generated (dry-run or --skip-bench)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        write_console("\nInterrupted.", error=True)
        raise SystemExit(130) from None
    except Exception as exc:
        write_console(f"\nERROR: {exc}", error=True)
        raise SystemExit(1) from None

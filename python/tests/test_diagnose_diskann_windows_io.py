from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "diagnose_diskann_windows_io.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_diskann_windows_io", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load diagnose_diskann_windows_io.py")
diagnose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnose
SPEC.loader.exec_module(diagnose)


FIELDS = (
    "mode",
    "random_access_hint",
    "queue_depth",
    "max_batch_size",
    "batch_gap_us",
    "actual_gap_us",
    "iops",
    "batch_count",
    "first_completion_ms",
    "batch_duration_ms",
    "completions_per_dequeue",
    "completed_reads",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "pending_ratio_pct",
)


def output_row(mode: str, iops: float) -> str:
    values = {
        "mode": mode,
        "random_access_hint": "on",
        "queue_depth": "0",
        "max_batch_size": "20",
        "batch_gap_us": "0",
        "actual_gap_us": "0.00",
        "iops": str(iops),
        "batch_count": "100",
        "first_completion_ms": "0.25",
        "batch_duration_ms": "0.55",
        "completions_per_dequeue": "1.60",
        "completed_reads": "20000",
        "avg_latency_ms": "0.30",
        "p50_latency_ms": "0.25",
        "p95_latency_ms": "0.50",
        "p99_latency_ms": "0.75",
        "pending_ratio_pct": "100.00",
    }
    return ",".join(values[field] for field in FIELDS)


def benchmark_output(iops: tuple[float, float, float, float]) -> list[str]:
    lines = ["DiskANN Windows IOCP microbenchmark", ",".join(FIELDS)]
    for mode, value in zip(diagnose.CACHED_HANDLE_ABBA_MODES, iops, strict=True):
        lines.append("[ INFO ] Opened file")
        lines.append(output_row(mode, value))
    return lines


def test_parse_args_enables_cached_handle_abba(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_diskann_windows_io.py",
            "--index-file",
            "index.bin",
            "--query-file",
            "query.txt",
            "--cached-handle-abba",
        ],
    )

    args = diagnose.parse_args()

    assert args.cached_handle_abba
    assert not args.full


def test_parse_args_rejects_cached_handle_abba_with_full(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_diskann_windows_io.py",
            "--index-file",
            "index.bin",
            "--query-file",
            "query.txt",
            "--cached-handle-abba",
            "--full",
        ],
    )

    with pytest.raises(SystemExit) as error:
        diagnose.parse_args()

    assert error.value.code == 2


def test_cached_handle_abba_parser_and_supported_verdict() -> None:
    rows = diagnose.cached_handle_abba_rows(
        benchmark_output((100.0, 50.0, 55.0, 110.0))
    )

    analysis = diagnose.analyze_cached_handle_abba(rows)

    assert [row["mode"] for row in rows] == list(diagnose.CACHED_HANDLE_ABBA_MODES)
    assert analysis.stable
    assert analysis.closed_geomean_iops == pytest.approx(math.sqrt(11_000.0))
    assert analysis.held_geomean_iops == pytest.approx(math.sqrt(2_750.0))
    assert analysis.held_retention_ratio == pytest.approx(0.5)
    assert analysis.verdict == "cached_handle_slowdown_supported"


def test_cached_handle_abba_parser_rejects_wrong_order() -> None:
    lines = benchmark_output((100.0, 50.0, 55.0, 110.0))
    first_row = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("cached_closed_a1,")
    )
    second_row = next(
        index for index, line in enumerate(lines) if line.startswith("cached_held_b1,")
    )
    lines[first_row], lines[second_row] = lines[second_row], lines[first_row]

    with pytest.raises(RuntimeError, match="A1/B1/B2/A2"):
        diagnose.cached_handle_abba_rows(lines)


def test_cached_handle_abba_parser_rejects_missing_row() -> None:
    lines = benchmark_output((100.0, 50.0, 55.0, 110.0))
    lines = [line for line in lines if not line.startswith("cached_held_b2,")]

    with pytest.raises(RuntimeError, match="A1/B1/B2/A2"):
        diagnose.cached_handle_abba_rows(lines)


def test_cached_handle_abba_parser_rejects_zero_iops() -> None:
    with pytest.raises(RuntimeError, match="finite and positive"):
        diagnose.cached_handle_abba_rows(benchmark_output((100.0, 0.0, 55.0, 110.0)))


def test_cached_handle_abba_unstable_pair_is_inconclusive() -> None:
    rows = diagnose.cached_handle_abba_rows(
        benchmark_output((100.0, 50.0, 52.0, 125.0))
    )

    analysis = diagnose.analyze_cached_handle_abba(rows)

    assert not analysis.stable
    assert analysis.closed_spread_ratio == pytest.approx(0.25)
    assert analysis.verdict == "inconclusive_unstable"


def test_cached_handle_abba_stable_small_effect_is_not_supported() -> None:
    rows = diagnose.cached_handle_abba_rows(
        benchmark_output((100.0, 95.0, 100.0, 105.0))
    )

    analysis = diagnose.analyze_cached_handle_abba(rows)

    assert analysis.stable
    assert analysis.held_retention_ratio > 0.8
    assert analysis.verdict == "cached_handle_slowdown_not_supported"


def test_cached_handle_abba_command_uses_focused_fixed_warmup_mode() -> None:
    command = diagnose.cached_handle_abba_command(
        Path("diskann_iocp_bench.exe"),
        Path("index.bin"),
        Path("trace.csv"),
        replay_seconds=10,
    )

    assert command == [
        Path("diskann_iocp_bench.exe"),
        "--file",
        Path("index.bin"),
        "--trace-file",
        Path("trace.csv"),
        "--duration",
        "10",
        "--cached-handle-abba",
    ]
    assert "--warmup" not in command
    assert "--reader-replay" not in command


def test_write_cached_handle_abba_results(tmp_path) -> None:
    rows = diagnose.cached_handle_abba_rows(
        benchmark_output((100.0, 50.0, 55.0, 110.0))
    )

    analysis = diagnose.write_cached_handle_abba_results(tmp_path, rows)

    with (tmp_path / "cached_handle_abba.csv").open(
        encoding="utf-8", newline=""
    ) as output:
        written_rows = list(csv.DictReader(output))
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert [row["mode"] for row in written_rows] == list(
        diagnose.CACHED_HANDLE_ABBA_MODES
    )
    assert analysis.verdict in summary
    assert "Held/closed throughput retention: **50.0%**" in summary
    assert "aligned 512 KiB reader buffer" in summary
    assert "one complete unmeasured trace warmup cycle" in summary
    assert "same buffered probe read" in summary

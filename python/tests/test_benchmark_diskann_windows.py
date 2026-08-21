from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_diskann_windows.py"
SPEC = importlib.util.spec_from_file_location("benchmark_diskann_windows", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_default_matrix_and_native_build_commands(tmp_path, monkeypatch):
    args = benchmark.parse_args([])
    assert args.train_file == benchmark.DEFAULT_TRAIN_FILE
    assert args.query_file == benchmark.DEFAULT_QUERY_FILE
    assert args.ground_truth_file == benchmark.DEFAULT_GROUND_TRUTH_FILE
    assert args.ground_truth_mode == "generate"
    assert args.repo_root == benchmark.DEFAULT_REPO_ROOT
    assert args.build_dir == benchmark.DEFAULT_BUILD_DIR
    assert tuple(args.precision) == ("fp32", "fp16")
    assert tuple(args.list_sizes) == (100, 300, 500)
    assert tuple(args.thread_counts) == (1, 2, 4)
    assert args.max_train_samples == 200_000
    assert args.disable_id_map is True
    assert len(args.precision) * len(args.list_sizes) * len(args.thread_counts) == 18

    repo_root = tmp_path / "repo"
    build_dir = tmp_path / "build"
    repo_root.mkdir()
    commands = []
    monkeypatch.setattr(benchmark, "executable", lambda *_args: "cmake")
    monkeypatch.setattr(
        benchmark,
        "run_simple",
        lambda command, **_kwargs: commands.append([str(part) for part in command]),
    )
    tools = benchmark.ensure_tools(
        repo_root,
        build_dir,
        {},
        skip_build=False,
        parallel_builds=8,
        dry_run=True,
    )
    assert commands[0][commands[0].index("-S") + 1] == str(repo_root)
    assert commands[0][commands[0].index("-B") + 1] == str(build_dir)
    assert commands[1][0:3] == ["cmake", "--build", str(build_dir)]
    assert set(tools) == {
        "local_builder",
        "recall_original",
        "bench_original",
    }


def test_generated_yaml_matches_document(tmp_path):
    train = tmp_path / "cohere train.vecs"
    query = tmp_path / "query.txt"
    ground_truth = tmp_path / "neighbors.txt"
    index = tmp_path / "diskann_fp16.index"
    build = benchmark.build_yaml(
        train_file=train,
        index_path=index,
        converter="CosineFp16Converter",
        build_threads=8,
        max_degree=32,
        builder_list_size=50,
        memory_limit=100.0,
        pq_chunks=384,
        max_train_samples=200_000,
        disable_id_map=True,
    )
    assert "ConverterName: CosineFp16Converter" in build
    assert "MetricName: Cosine" in build
    assert "DisableIdMap: true" in build
    assert "ThreadCount: 8" in build
    assert "max_degree: !!int 32" in build
    assert "list_size: !!int 50" in build
    assert "max_pq_chunk_num: !!int 384" in build
    assert "max_train_sample_count: !!int 200000" in build
    assert "\\" not in build

    search = benchmark.search_yaml(
        index_path=index,
        query_file=query,
        ground_truth_file=ground_truth,
        recall_log_dir=tmp_path / "recall",
        top_k="1,10,50",
        recall_gt_count=100,
        recall_threads=16,
        bench_threads=4,
        bench_seconds=30,
        bench_iterations=10_000_000,
        cache_nodes=10_000,
        list_size=300,
    )
    assert "TopK: 1,10,50" in search
    assert "RecallGTCount: 100" in search
    assert "BenchThreadCount: 4" in search
    assert "cache_node_num: !!int 10000" in search
    assert "list_size: !!int 300" in search


def test_parse_recall_and_benchmark_log(tmp_path):
    recall_log = tmp_path / "recall.log"
    recall_log.write_text(
        "\n".join(
            [
                "Load external ground truth file[neighbors.txt] done!",
                "Internal ground truth file NOT used since external ground truth file has been loaded",
                "Load index done!",
                "Process query: 1000",
                "Recall@1: 93.1",
                "Recall@10: 94.26",
                "Recall@50: 91.398",
                "Recall done.",
            ]
        ),
        encoding="utf-8",
    )
    recall = benchmark.parse_recall_result(
        precision="fp32",
        list_size=100,
        log_path=recall_log,
        metrics=benchmark.ProcessMetrics(wall_seconds=5.0, peak_rss_mib=450.0),
        expected_query_count=1000,
        external_ground_truth=True,
    )
    assert recall.recall_at_1_pct == 93.1
    assert recall.recall_at_10_pct == 94.26
    assert recall.recall_at_50_pct == 91.398

    bench_log = tmp_path / "bench.log"
    bench_log.write_text(
        "\n".join(
            [
                "Load index done!",
                "Process query: 7053, total process time: 30328ms, duration: 30000ms, max: 12ms, min: 2ms",
                "Avg latency: 4.3ms qps: 235.1",
                "25 Percentile: 3.8 ms",
                "50 Percentile: 4.1 ms",
                "75 Percentile: 4.6 ms",
                "90 Percentile: 5.1 ms",
                "95 Percentile: 5.5 ms",
                "99 Percentile: 6.4 ms",
                "Bench done.",
            ]
        ),
        encoding="utf-8",
    )
    result = benchmark.parse_search_result(
        precision="fp32",
        list_size=100,
        threads=1,
        recall=recall,
        recall_log_path=recall_log,
        bench_log_path=bench_log,
        metrics=benchmark.ProcessMetrics(
            wall_seconds=32.0,
            peak_rss_mib=468.0,
            read_operations=80_000,
            read_iops=2_600.0,
            read_mb_per_second=12.0,
        ),
        bench_seconds=30,
    )
    assert result.qps == 235.1
    assert result.avg_latency_ms == 4.3
    assert result.peak_rss_mib == 468.0
    assert result.reads_per_query == pytest.approx(80_000 / 7053)


def test_peak_rss_uses_windows_peak_working_set():
    memory = benchmark.PROCESS_MEMORY_COUNTERS_EX()
    memory.WorkingSetSize = 120 * benchmark.MIB
    memory.PeakWorkingSetSize = 468 * benchmark.MIB
    io = benchmark.IO_COUNTERS()
    io.ReadOperationCount = 123
    io.ReadTransferCount = 456
    sample = benchmark.process_sample_from_counters(memory, io)
    assert sample.current_rss_bytes == 120 * benchmark.MIB
    assert sample.peak_rss_bytes / benchmark.MIB == 468
    assert sample.read_operations == 123


def make_results():
    builds = [
        benchmark.BuildResult(
            precision=precision,
            index_path=f"{precision}.index",
            index_size_gib=4.18 if precision == "fp32" else 2.27,
            train_seconds=200.0,
            build_seconds=210.0,
            dump_seconds=10.0,
            wall_seconds=430.0,
            peak_rss_mib=2048.0,
            log_path=f"build_{precision}.log",
        )
        for precision in ("fp32", "fp16")
    ]
    recalls = [
        benchmark.RecallResult(
            precision=precision,
            list_size=list_size,
            recall_at_1_pct=93.1,
            recall_at_10_pct=94.26,
            recall_at_50_pct=91.398,
            query_count=1000,
            wall_seconds=5.0,
            peak_rss_mib=500.0,
            log_path=f"recall_{precision}_{list_size}.log",
        )
        for precision in ("fp32", "fp16")
        for list_size in (100, 300, 500)
    ]
    recall_map = {(row.precision, row.list_size): row for row in recalls}
    searches = [
        benchmark.SearchResult(
            precision=precision,
            list_size=list_size,
            threads=threads,
            recall_at_1_pct=recall_map[(precision, list_size)].recall_at_1_pct,
            recall_at_10_pct=recall_map[(precision, list_size)].recall_at_10_pct,
            recall_at_50_pct=recall_map[(precision, list_size)].recall_at_50_pct,
            qps=235.1,
            avg_latency_ms=4.3,
            p50_latency_ms=4.1,
            p95_latency_ms=5.5,
            p99_latency_ms=6.4,
            min_latency_ms=2.0,
            max_latency_ms=12.0,
            query_count=7053,
            duration_ms=30_000,
            peak_rss_mib=468.0 + threads,
            recall_peak_rss_mib=500.0,
            process_read_iops=2600.0,
            process_read_mb_per_second=12.0,
            reads_per_query=11.3,
            recall_log_path=recall_map[(precision, list_size)].log_path,
            bench_log_path=f"bench_{precision}_{list_size}_{threads}.log",
        )
        for precision in ("fp32", "fp16")
        for list_size in (100, 300, 500)
        for threads in (1, 2, 4)
    ]
    return builds, recalls, searches


def test_report_contract_and_missing_required_metrics(tmp_path):
    builds, recalls, searches = make_results()
    metadata = {
        "timestamp": "2026-08-18T00:00:00+08:00",
        "git_sha": "189282b",
        "git_branch": "feat/diskann-support-windows",
        "git_dirty": False,
        "server_label": "ecs.g9i.4xlarge",
        "platform": "Windows",
        "processor": "Intel",
        "logical_cpu_count": 16,
        "python": "3.12",
        "io_backend": "windows_overlapped",
        "io_backend_description": "IOCP",
        "ground_truth_file": "neighbors.txt",
        "query_count": 1000,
        "parameters": {
            "precision": ["fp32", "fp16"],
            "list_sizes": [100, 300, 500],
            "thread_counts": [1, 2, 4],
            "build_threads": 8,
            "max_degree": 32,
            "builder_list_size": 50,
            "pq_chunks": 384,
            "memory_limit": 100.0,
            "cache_nodes": 10_000,
            "bench_top_k": 50,
        },
    }
    benchmark.validate_result_contract(
        metadata,
        builds,
        recalls,
        searches,
        require_recalls=True,
        require_searches=True,
    )
    benchmark.write_outputs(tmp_path, metadata, builds, recalls, searches)
    with (tmp_path / "results.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert {
        "precision",
        "list_size",
        "threads",
        "recall_at_1_pct",
        "recall_at_10_pct",
        "recall_at_50_pct",
        "qps",
        "peak_rss_mib",
    }.issubset(rows[0])
    document = (tmp_path / "document_results.md").read_text(encoding="utf-8")
    assert "Recall@1 %" in document
    assert "QPS (4 thread)" in document
    assert "Peak RSS MiB (4 thread)" in document

    searches[0].qps = 0
    with pytest.raises(ValueError, match="QPS and Peak RSS"):
        benchmark.validate_result_contract(
            metadata,
            builds,
            recalls,
            searches,
            require_recalls=True,
            require_searches=True,
        )


def test_reused_recall_requires_matching_provenance():
    _builds, recalls, _searches = make_results()
    metadata = {
        "git_sha": "189282b",
        "git_dirty": False,
        "query_file": "query.txt",
        "query_count": 1000,
        "query_file_signature": {"path": "query.txt", "size": 10, "mtime_ns": 1},
        "ground_truth_file": "neighbors.txt",
        "ground_truth_file_signature": {
            "path": "neighbors.txt",
            "size": 20,
            "mtime_ns": 2,
        },
        "index_files": {
            "fp32": {"path": "fp32.index", "size": 30, "mtime_ns": 3},
            "fp16": {"path": "fp16.index", "size": 20, "mtime_ns": 4},
        },
        "parameters": {
            "precision": ["fp32", "fp16"],
            "list_sizes": [100, 300, 500],
            "cache_nodes": 10_000,
            "top_k": "1,10,50",
            "ground_truth_mode": "external",
            "dimension": 768,
            "ground_truth_k": 100,
        },
    }
    previous = {**metadata, "parameters": dict(metadata["parameters"])}
    benchmark.validate_reused_recall(previous, metadata, recalls)

    previous["query_file"] = "different-query.txt"
    with pytest.raises(ValueError, match="provenance changed"):
        benchmark.validate_reused_recall(previous, metadata, recalls)

    previous["query_file"] = metadata["query_file"]
    previous["git_dirty"] = True
    with pytest.raises(ValueError, match="provenance changed"):
        benchmark.validate_reused_recall(previous, metadata, recalls)


def test_empty_csv_removes_stale_results(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("stale\n", encoding="utf-8")
    benchmark.write_csv(path, [])
    assert not path.exists()

"""Compare reused one-worker results with newly measured 2/4/8 workers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_diskann_paced_io import aggregate, evaluate_probe
from benchmark_diskann_concurrent_io import validate_concurrent_query
from summarize_diskann_controlled_io import EXPECTED


BASELINE = Path(__file__).resolve().parents[1] / "benchmarks/diskann-controlled-io/results/2026-11-05/all-platform-results.json"


def validate_report(report):
    baseline_bytes = BASELINE.read_bytes()
    baseline = {r["label"]: r for r in json.loads(baseline_bytes)}
    if (report["kind"] != "concurrent-paced-native-file-io" or report["label"] not in EXPECTED or
            report["new_query_threads"] != [2, 4, 8] or not report["baseline_reused"] or
            report["baseline_file_sha256"] != hashlib.sha256(baseline_bytes).hexdigest() or
            report["baseline"] != baseline[report["label"]] or
            report["fixture"] != report["baseline"]["fixture"] or report["queue_depth"] != 8 or
            report["trials"] != 3 or report["repetitions"] != 3):
        raise ValueError("Concurrent protocol or reused one-worker baseline changed")
    for rate, probe in report["probes"].items():
        if evaluate_probe(probe["raw"], report["label"], int(rate)) != probe["verification"]:
            raise ValueError("Probe result differs from raw measurements")
    trials = 0
    incomplete = []
    for rate in report["requested_profiles"]:
        profile = report["profiles"].get(str(rate), {})
        baseline_profile = report["baseline"]["profiles"].get(str(rate), {})
        reused = dict(baseline_profile, reused=True)
        if profile and profile["threads"].get("1") != reused:
            raise ValueError("One-worker data was rerun or changed")
        for workers in (2, 4, 8):
            entry = profile.get("threads", {}).get(str(workers), {})
            if not entry.get("eligible"):
                if rate in (0, report["common_iops"]):
                    incomplete.append(f"{report['label']}/{rate}/{workers}")
                continue
            complete = True
            for algorithm in ("hnsw", "diskann"):
                rows = entry["raw_trials"][algorithm]
                if len(rows) != 3:
                    complete = False
                for query in rows:
                    validate_concurrent_query(query, algorithm, report["fixture"], rate, report["label"],
                                              workers, profile["bytes_per_second_limit"], report["queue_depth"])
                    if query["queries"] != 1000 or query["repetitions"] != 3:
                        raise ValueError("Total held-out workload changed across thread counts")
                    trials += 1
                if len(rows) == 3 and entry.get("medians", {}).get(algorithm) != aggregate(rows):
                    raise ValueError("Reported concurrent median differs from raw trials")
            if not complete:
                incomplete.append(f"{report['label']}/{rate}/{workers}")
    if not report.get("measurement_complete"):
        incomplete.append(report["label"] + "/not-complete")
    return trials, incomplete


def build_report(root):
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("concurrent-results.json"))]
    if not reports:
        raise ValueError("No concurrent measurements")
    labels = [r["label"] for r in reports]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate concurrent environment")
    first = reports[0]
    failures = sorted(EXPECTED - set(labels))
    measured_trials = 0
    for report in reports:
        if any(report[key] != first[key] for key in (
                "fixture", "requested_profiles", "new_query_threads", "queue_depth", "common_iops")):
            raise ValueError("Cannot mix concurrent fixtures or total I/O budgets")
        if (report["host"]["commit"] != first["host"]["commit"] or
                report["host"]["run_url"] != first["host"]["run_url"]):
            raise ValueError("Cannot mix concurrent benchmark revisions/runs")
        count, incomplete = validate_report(report)
        measured_trials += count
        failures.extend(incomplete)
    dataset = first["fixture"]["dataset"]
    lines = ["# HNSW / DiskANN 八平台并发实测", "",
             f"同一份 {dataset['vectors']:,} 条、{dataset['dimension']} 维 SIFT FP32 数据与已训练索引，固定查询参数。",
             "本轮只新增 2、4、8 个查询工作线程；1 线程直接沿用先前复核通过的结果，未重跑。"
             "1 线程来自此前的 CI 实例，不能据此把相对它的差异全部归因于线程数。", "",
             "每个进程共享一个索引。1,000 条评估查询合计重复 3 遍，共 3,000 次调用，按线程均匀分摊，"
             "不随线程数增加总查询量。先预热一次工作负载，再在三个独立进程试验中测量并取中位数。", "",
             "IOPS、带宽和最多 8 个在途读取由所有线程共享。P95 为所有线程实际搜索调用耗时的分位数，"
             "包含限速与 I/O 队列等待；QPS 为总完成查询数除以整个进程计时区间的真实经过时间。", "",
             "RSS 是所有工作线程及其查询缓冲区仍存活时的进程总量；Windows 使用工作集，单位为十进制 MB。"
             "HNSW 已预热驻留内存；DiskANN 使用原生文件读取，缓存条件逐行列明。", "",
             f"新增测试来源：{first['host']['run_url']}；提交 `{first['host']['commit']}`。"
             f"1 线程来源：{first['baseline']['host']['run_url']}；提交 `{first['baseline']['host']['commit']}`。", "",
             "数值按 **HNSW → DiskANN** 排列。† 表示宿主未通过该档位的吞吐资格；仍列出相同配置上限下的真实结果与实际 IOPS。", ""]
    for rate in first["requested_profiles"]:
        lines += [f"## 进程总上限 {rate:,} IOPS" if rate else "## 原生不限速参考（进程总读取深度仍为 8）", ""]
        for simulator in (False, True):
            lines += ["移动端模拟器：" if simulator else "桌面 CI：", "",
                      "| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
            for report in sorted(reports, key=lambda r: r["label"]):
                mobile = "emulator" in report["label"] or "simulator" in report["label"]
                if mobile != simulator:
                    continue
                for workers in (1, 2, 4, 8):
                    profile = report["profiles"].get(str(rate), {})
                    entry = profile.get("threads", {}).get(str(workers), {})
                    name = report["label"] + (" **模拟器**" if mobile else "")
                    worker_name = "1（沿用）" if workers == 1 else str(workers)
                    if not entry.get("eligible") or not entry.get("medians"):
                        lines.append(f"| {name} | {worker_name} | — | — | — | — | 未完成或该可选档位未达标 |")
                        continue
                    h, d = entry["medians"]["hnsw"], entry["medians"]["diskann"]
                    mark = " †" if workers > 1 and not entry.get("host_qualified", True) else ""
                    lines.append(f"| {name}{mark} | {worker_name} | {h['steady_rss_bytes']/1e6:.2f} → {d['steady_rss_bytes']/1e6:.2f} | "
                                 f"{h['recall_at_10']:.2%} → {d['recall_at_10']:.2%} | {h['p95_ms']:.3f} → {d['p95_ms']:.3f} | "
                                 f"{h['qps']:,.2f} → {d['qps']:,.2f} | {d['cache_policy']} / {d['achieved_iops']:,.0f} |")
            lines += [""]
    lines += ["## 验证范围", "",
              "- 本轮新增的 2/4/8 线程共用各自平台的同一个 CI 实例；各操作系统的 CPU、内存及宿主存储仍不同。",
              "- Linux/Android 核对 OS 存储读字节；Android 为模拟器未加密目录的控制实验。",
              "- Apple 为按范围驱逐页缓存，iOS 使用宿主对模拟器进程的计数；Windows 使用 FILE_FLAG_NO_BUFFERING。",
              "- Windows 物理设备读及所有平台的设备/虚拟化宿主缓存未完全独立排除。",
              f"- 本轮复核 {measured_trials} 次新增查询进程试验；1 线程历史记录另存于每个平台的 baseline 字段。", ""]
    if failures:
        lines += ["未完成：" + ", ".join(failures), ""]
    return "\n".join(lines), reports, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown, reports, failures = build_report(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "concurrent-comparison.md").write_text(markdown)
    (args.output / "all-platform-concurrent-results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(markdown)
    if failures:
        raise SystemExit("Concurrent measurement incomplete; partial data retained")


if __name__ == "__main__":
    main()

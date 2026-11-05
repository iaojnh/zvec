"""Build publishable HNSW/DiskANN tables from verified native pacing runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from summarize_diskann_controlled_io import EXPECTED


def build_report(root):
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("paced-results.json"))]
    labels = [r["label"] for r in reports]
    if len(set(labels)) != len(labels) or any(label not in EXPECTED for label in labels):
        raise ValueError("Duplicate/unknown environment")
    if not reports:
        raise ValueError("No native pacing results")
    first = reports[0]
    for report in reports:
        if (report["kind"] != "paced-native-file-io" or report["fixture"] != first["fixture"] or
                report["common_iops"] != first["common_iops"] or report["queue_depth"] != first["queue_depth"] or
                report["requested_profiles"] != first["requested_profiles"] or
                report.get("probe_operations") != first.get("probe_operations") or
                report.get("probe_warmup_operations") != first.get("probe_warmup_operations") or
                report["host"]["commit"] != first["host"]["commit"] or
                report["host"]["run_url"] != first["host"]["run_url"] or
                report["trials"] != first["trials"] or report["repetitions"] != first["repetitions"]):
            raise ValueError("Cannot combine different fixtures or pacing conditions")
    dataset = first["fixture"]["dataset"]
    lines = [
        "# DiskANN 全平台真实 I/O 限速实测", "",
        f"统一使用 {dataset['vectors']:,} 条、{dataset['dimension']} 维 FP32 SIFT 数据及同一份已训练索引，固定查询参数。单并发 C++ 核心接口；"
        f"每个索引预热 {dataset['evaluation_queries']:,} 条查询后，计时执行 {dataset['evaluation_queries']:,} 条查询 × {first['repetitions']} 遍，"
        f"{first['trials']} 个独立进程试验取中位数。", "",
        "P95 按真实搜索调用耗时统计，QPS 按完整计时区间的实际经过时间计算；原生文件读取与限速等待均计入。"
        "HNSW 预热后驻留内存，DiskANN 保留生产文件读取路径，不预载完整图和向量。", "",
        "RSS 为查询进程内存，Windows 为工作集，单位 MB（十进制）；不包含未映射的系统文件缓存。"
        "CPU、内存和存储仍为各 CI runner 实际配置；受控 IOPS 是已验证的速率上限。"
        "移动端均为模拟器，结果不代表手机真机。", "",
        f"来源：{first['host']['run_url']}；提交 `{first['host']['commit']}`。"
        f"每个探针档位先预热 {first.get('probe_warmup_operations', 0):,} 次，再计时读取 "
        f"{first.get('probe_operations', 2048):,} 次（每次 4 KiB）。IOPS 限制针对原生文件读请求，"
        "物理设备操作可能被固件或虚拟化层合并或放大。", "",
        "下表所有数值按 **HNSW → DiskANN** 排列。", "",
    ]
    def table(rate, simulator):
        selected = [r for r in reports if ("emulator" in r["label"] or "simulator" in r["label"]) == simulator]
        out = ["| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |",
               "| --- | ---: | ---: | ---: | ---: | --- |"]
        for report in sorted(selected, key=lambda r: r["label"]):
            profile = report["profiles"].get(str(rate), {})
            name = report["label"] + (" **模拟器**" if simulator else "")
            if not profile.get("eligible") or not profile.get("medians"):
                out.append(f"| {name} | — | — | — | — | 未通过此档位验证，留空 |")
                continue
            h, d = profile["medians"]["hnsw"], profile["medians"]["diskann"]
            out.append(f"| {name} | {h['steady_rss_bytes']/1e6:.2f} → {d['steady_rss_bytes']/1e6:.2f} | "
                       f"{h['recall_at_10']:.2%} → {d['recall_at_10']:.2%} | {h['p95_ms']:.3f} → {d['p95_ms']:.3f} | "
                       f"{h['qps']:,.1f} → {d['qps']:,.1f} | {d['cache_policy']} / {d['achieved_iops']:,.0f} |")
        return out
    for rate in first["requested_profiles"]:
        lines += [f"## {rate:,} IOPS 上限" if rate else "## 原生不限速参考", ""]
        if rate:
            lines += [f"最多 {first['queue_depth']} 个请求并发；该档位须先通过真实速率、半速对照和缓存验证。", ""]
        else:
            lines += ["各平台原生磁盘不同，本组展示各自 CI 环境下的性能，不作为操作系统排名。", ""]
        lines += table(rate, False) + ["", "移动端模拟器：", ""] + table(rate, True) + [""]
    lines += ["## 缓存与验证范围", "",
              "- Linux/Android：结合预热缓冲读取对照与 OS 存储读字节验证。Android 使用 CI 模拟器的未加密控制目录。",
              "- macOS/iOS：生产读取器按请求范围驱逐缓存，并核对 OS 存储读字节；iOS 由宿主观察模拟器进程计数。",
              "- Windows：原生 FILE_FLAG_NO_BUFFERING；进程 I/O 计数为逻辑传输量，未独立验证物理设备缓存。",
              "- 设备及虚拟化宿主层的缓存不能据此全部排除，数据仅对应记录的 CI 环境。", ""]
    missing = sorted(EXPECTED - set(labels))
    failed = [r["label"] for r in reports if not r.get("publication_ready")]
    if missing or failed:
        lines += ["未完成或未通过公共档位验证：" + ", ".join(missing + failed), ""]
    return "\n".join(lines), reports, missing + failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown, reports, failures = build_report(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "publication-comparison.md").write_text(markdown)
    (args.output / "all-platform-results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(markdown)
    if failures:
        raise SystemExit("Incomplete/failed public tier; partial report preserved")


if __name__ == "__main__":
    main()

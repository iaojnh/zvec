"""Report the complete mmap-prefetch cohort, retaining low-memory regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
root = a.root.resolve()


def read(path):
    return json.loads(path.read_text())


manifest = read(root / "manifest.json")
summary = read(root / "current-summary.json")
validation = read(root / "current-validation.json")
assert validation["complete"] and validation["valid_runs"] == 18
assert read(root / "regression-validation.json")["passed"] == 97
assert read(root / "postflight.json")["complete"]
by_key = {(v["limit_mib"], v["mode"]): v for v in summary}
prior = root.parent / "ivf-mmap-bp-current-linux-cgroup-20261110"
reference = read(root / "inputs/reference-zvec.json")
context = []
for limit, repeats in [(1024, [1, 2]), (2048, [1, 2, 3])]:
    runs = []
    for repeat in repeats:
        name = f"r{repeat}-m{limit}-mmap"
        v = read(prior / "raw" / (name + ".json"))
        m = read(prior / "meta" / (name + ".json"))
        assert v["complete"] and m["validation_complete"] and m["returncode"] == 0
        assert v["query_ids"] == reference["query_ids"] and v["result_ids"] == reference["result_ids"]
        assert v["recall_at_10"] == .9645 and v["nprobe"] == 12
        assert m["cgroup_after"]["memory.max"] == limit << 20
        assert m["cgroup_after"]["memory.swap.max"] == m["cgroup_after"]["memory.swap.current"] == 0
        assert m["cgroup_after"]["memory.events"]["oom_kill"] == 0
        assert all(x["resident_after"] == 0 for x in m["cold_inputs"])
        history = read(prior / "commands" / (name + ".isolation.json"))
        assert all(x["profiles"]["default"] == "Stopped" for x in history)
        runs.append(dict(name=name, qps=v["qps"], user_ms=v["user_cpu_s"] * 2.5,
                         system_ms=v["system_cpu_s"] * 2.5, major_faults=v["os_pageins"],
                         minor_faults=v["os_minor_faults"], read_gib=v["os_disk_read_bytes"] / 2**30,
                         raw_sha256=hashlib.sha256((prior / "raw" / (name + ".json")).read_bytes()).hexdigest()))
    context.append(dict(limit_mib=limit, mode="mmap_no_hint", repeats=len(runs), runs=runs,
                        medians={k: statistics.median(v[k] for v in runs) for k in
                                 ["qps", "user_ms", "system_ms", "major_faults", "minor_faults", "read_gib"]},
                        strict_same_batch_with_fresh_prefetch=False,
                        note="Recent integration-base audit, partial1GiB scope retained; not a new same-cohort A/B."))
(root / "recent-no-hint-context.json").write_text(json.dumps(context, indent=2) + "\n")
lines = ["正式 mmap CPU 预取：1/2/4 GiB 重测", "2026-11-10", "",
"结论：4 GiB 下 mmap 与 BP 接近；1/2 GiB 下直接扩大 CPU 预取范围出现回退。",
"该改动保存在独立分支，未合入集成分支。不能把满驻留场景的收益外推到内存不足场景。", "",
"代码：" + manifest["product_commit"],
"仅修改两处行数据评分前的 CPU cache line 提示条件，覆盖 mmap 稳定连续地址；BP 已有提示保留。",
"无操作系统预取、缓存容量、聚类中心、路由、距离计算或召回参数调整。", "",
"共同条件：1M×768 FP32 cosine，固定160中心，nprobe12，原200条预热+原400条正式查询。",
"mmap/BP 同一个正式二进制，三个内存限额各三轮交错运行，18次结果有序Top10完全一致，Recall@10=96.45%。",
"限额为 cgroup v2 总内存（匿名页、文件页缓存及内核计费），单guest CPU1，swap0。",
"默认另一台虚拟机全程停止；保留用户授权并行的宿主机 DiskANN 作业。", "",
"GiB | mmap QPS | BP QPS | BP/mmap | mmap三轮 | BP三轮"]
for limit in [1024, 2048, 4096]:
    mm, bp = by_key[limit, "mmap"], by_key[limit, "bp"]
    lines.append(f"{limit//1024} | {mm['qps']:.4f} | {bp['qps']:.4f} | {bp['qps']/mm['qps']:.3f} | "
                 + ", ".join(f"{x:.4f}" for x in mm["qps_values"]) + " | "
                 + ", ".join(f"{x:.4f}" for x in bp["qps_values"]))
lines += ["", "GiB | 路径 | RSS MiB | cgroup峰值 MiB | 用户态ms/query | 系统态ms/query | 读盘GiB/400条 | major | minor"]
for v in sorted(summary, key=lambda v: (v["limit_mib"], v["mode"])):
    lines.append(f"{v['limit_mib']//1024} | {v['mode']} | {v['rss_mib']:.2f} | {v['cgroup_peak_mib']:.2f} | "
                 f"{v['user_ms']:.5f} | {v['system_ms']:.5f} | {v['read_gib']:.6f} | "
                 f"{v['major_faults']:.0f} | {v['minor_faults']:.0f}")
lines += ["", "最近未加 mmap 预取的集成版本记录（跨批次背景，非本轮严格A/B）："]
for v in context:
    med = v["medians"]
    lines.append(f"{v['limit_mib']//1024} GiB，{v['repeats']}次：QPS {med['qps']:.4f}，"
                 f"用户态{med['user_ms']:.4f}ms/query，系统态{med['system_ms']:.4f}ms/query，"
                 f"读盘{med['read_gib']:.5f}GiB，minor {med['minor_faults']:.0f}。")
lines += ["", "4 GiB 先前专门控制实验，三轮中位数：默认mmap98.5837、等同CPU提示mmap157.2909、BP154.0079 QPS。",
"这一严格控制实验对应约59.55%的mmap提升；不能与本轮不同批次直接混算代码收益。",
"4 GiB mmap前后读盘都为5MiB、major0、minor136；CPU路径对齐解释了此前大部分差距。", "",
"1/2 GiB 的回退证据：三轮均需保留，不能只选4 GiB的改进数据。",
"与最近未加预取记录相比，低内存读盘量和缺页规模接近，用户态CPU时间却明显增加。",
"这与大量非驻留页上的逐cache-line提示带来额外地址转换/页表遍历开销相符，但具体微架构原因尚未通过硬件计数器确认。",
"在证明低内存场景的开销受控之前，不把无条件mmap预取合入集成分支。", "",
"验证：97项回归通过，含mmap奇数维度、行尾不完整块、过滤/无过滤和Searcher/Streamer；macOS语法检查通过。",
"构建沿用冻结基线对象，重编译存储、IVFEntity、Searcher、Streamer、接口和相同基准适配器；未宣称全项目CI通过。",
"索引、查询、真值、执行文件及镜像前后哈希已核对，18次cgroup和冷启动检查均通过。",
"五读者更新表见同级 ivf-five-readers-mmap-prefetch-overview-20261110/report.txt；其SPANN/Faiss/LanceDB列仍为历史批次。"]
(root / "report.txt").write_text("\n".join(lines) + "\n")
print(root / "report.txt")

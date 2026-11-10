"""Replace the two zvec columns with a fresh, interleaved mmap/BP cohort."""
import argparse
import hashlib
import json
from pathlib import Path


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
p.add_argument("--out", type=Path)
a = p.parse_args()
base = a.results.resolve()
current = base / "ivf-mmap-prefetch-linux-cgroup-20261110"
previous = base / "ivf-five-readers-optimized-overview-20261110"
out = a.out or base / "ivf-five-readers-mmap-prefetch-overview-20261110"
out.mkdir(parents=True, exist_ok=True)
evidence = {}


def read(path):
    evidence[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


manifest = read(current / "manifest.json")
validation = read(current / "current-validation.json")
assert validation["complete"] and validation["valid_runs"] == 18
assert validation["ordered_top10_identical"] and validation["recall"] == .9645
assert read(previous / "validation.json")["complete"]
for name, digest in read(previous / "source-checksums.json").items():
    assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest
    evidence[name] = digest
old = read(previous / "overview.json")
prior_manifest = read(base / "ivf-capacity-reuse-linux-cgroup-20261110/manifest.json")
for key in ["input_sha256", "data_shape", "metric", "centroids_sha256", "fingerprint",
            "query_image_id", "query_cpuset", "swap_mib", "kernel"]:
    assert manifest[key] == prior_manifest[key]
index_provenance = read(base / "ivf-linux-cgroup-20261110/provenance.json")
assert manifest["file_sha256"]["/work/data/zvec-fixed.index"] == index_provenance["files"]["/work/data/zvec-fixed.index"]["sha256"]
assert read(current / "inputs/reference-zvec.json") == read(base / "ivf-capacity-reuse-linux-cgroup-20261110/inputs/reference-zvec.json")
fresh = read(current / "current-summary.json")
assert len(fresh) == 6 and all(x["repeats"] == 3 for x in fresh)
points = [dict(v, source_results=str(previous.relative_to(base)), cohort="previous_peer_batches")
          for v in old if v["engine"] in ["spann", "faiss", "lancedb"]]
for row in fresh:
    v = dict(row)
    v.update(engine="zvec" if v.pop("mode") == "bp" else "zvec_mmap", recall=.9645,
             nprobe=12, batch="mmap_prefetch_formal", cohort="fresh_interleaved_zvec",
             source_product_commit=manifest["product_commit"],
             source_results=str(current.relative_to(base)), kind="measured_three_repeat_median")
    points.append(v)
order = ["zvec_mmap", "zvec", "spann", "faiss", "lancedb"]
points.sort(key=lambda v: (v["limit_mib"], order.index(v["engine"])))
assert len(points) == 15
by_key = {(v["limit_mib"], v["engine"]): v for v in points}
gaps = []
for limit in [1024, 2048, 4096]:
    bp, mm = [by_key[limit, engine] for engine in ["zvec", "zvec_mmap"]]
    gaps.append(dict(limit_mib=limit, engines=["zvec", "zvec_mmap"],
                     bp_over_mmap=bp["qps"] / mm["qps"], strict_same_batch=True))
    for engine in ["spann", "faiss", "lancedb"]:
        peer = by_key[limit, engine]
        gaps.append(dict(limit_mib=limit, peer=engine, strict_same_batch=False,
                         bp_over_peer=bp["qps"] / peer["qps"],
                         mmap_over_peer=mm["qps"] / peer["qps"]))
save("overview.json", points)
save("gaps.json", gaps)
save("historical-zvec-points.json", [v for v in old if v["engine"] in ["zvec", "zvec_mmap"]])
save("measured-faiss-endpoints.json", read(previous / "measured-faiss-endpoints.json"))
save("validation.json", dict(complete=True, rows=15, fresh_zvec_raw_runs=18,
     prior_evidence_hashes_rechecked=True, original400_ordered_results_equal=True,
     shared_data_and_cgroup_settings=True, same_batch_mmap_and_bp=True,
     same_batch_all_five=False, peers_rerun=False, host_isolation_guaranteed=False,
     faiss_qps_interpolated_to_recall=.9645, faiss_shared_centroids=False))
save("manifest.json", dict(status="complete", date="2026-11-10", product_commit=manifest["product_commit"],
     integration_status=manifest["integration_status"], target_recall=.9645, queries=400, warmup=200,
     source_roots={"fresh_zvec": str(current), "prior_five_reader_overview": str(previous)},
     note="Both zvec columns are freshly interleaved. Peer columns retain their previous cohorts. No cross-batch normalization."))
save("source-checksums.json", evidence)

lines = ["mmap CPU 预取后的五种读取路径：1/2/4 GiB", "2026-11-10", "",
"结论：4 GiB 下 mmap 与 BP 接近；1/2 GiB 预取出现回退，当前改动未合入集成分支。",
"本轮只重新测试 zvec mmap 和 BP；两者使用同一个正式二进制，三轮交错执行。",
"SPANN、Faiss、LanceDB 沿用此前有效三轮结果，因此与它们的倍率属于跨批次数值比较。",
"原五读者总表保留，新表不覆盖历史数值。", "",
"共同工作负载：1M×768 FP32 cosine，160 中心，Top10，原200条预热和400条正式查询。",
"zvec/SPANN/LanceDB 使用同一批160中心、12条候选链，Recall@10=96.45%，有序Top10相同。",
"Faiss 独立训练160中心；QPS按p10/p11的每查询耗时插值到96.45%召回率。",
"限制是 cgroup v2 总内存（含匿名页、文件缓存、内核计费），单guest CPU1，swap0。",
"宿主机保留用户授权并行的 DiskANN 作业；默认另一台虚拟机全程保持停止。", "",
"QPS（三轮中位数）：", "GiB | zvec mmap 新预取 | zvec BP 同轮 | SPANN 旧批次 | Faiss* 旧批次 | LanceDB 旧批次"]
for limit in [1024, 2048, 4096]:
    lines.append(str(limit // 1024) + " | " + " | ".join(f"{by_key[limit, e]['qps']:.2f}" for e in order))
lines.extend(["", "zvec 三轮原始 QPS：", "GiB | 路径 | 三轮 QPS"])
for v in points:
    if v["engine"] in ["zvec", "zvec_mmap"]:
        lines.append(f"{v['limit_mib']//1024} | {v['engine']} | " + ", ".join(f"{x:.4f}" for x in v["qps_values"]))
lines.extend(["", "RSS、测量阶段读盘量、p95（各取三轮中位数）：",
              "GiB | 路径 | RSS MiB | cgroup峰值 MiB | 读盘 GiB/400查询 | p95 ms"])
for v in points:
    metrics = v.get("measured_p11_metrics", v)
    label = v["engine"] + (" p11实际96.575%" if v["engine"] == "faiss" else "")
    lines.append(f"{v['limit_mib']//1024} | {label} | {metrics['rss_mib']:.2f} | "
                 f"{metrics['cgroup_peak_mib']:.2f} | {metrics['read_gib']:.6f} | {metrics['p95_ms']:.3f}")
lines.extend(["", "预取修改与验证：",
"把 BP 已有的 CPU cache line 提示扩展到 mmap 稳定连续的普通行数据；不增加操作系统读盘预取或缓存额度。",
"打包量化格式和分散跨度仍不走这条提示路径。过滤/无过滤、尾块等新增回归及原有回归共97项通过，macOS语法检查通过。",
"正式代码提交：" + manifest["product_commit"],
"集成状态：" + manifest["integration_status"], "",
"对结论的限制：",
"4 GiB 数据几乎完全驻留时，mmap 预取可缩小此前 CPU 阶段的差距。",
"1/2 GiB 下必须同时查看用户态CPU和缺页，不能把4 GiB的收益外推到内存不足场景。",
"Faiss的RSS/IO/p95列为p11实际测量，不能与96.45%的插值QPS当作同一实际运行点。",
"跨批次差值不能全部归因于代码；同轮mmap/BP对照才具有当前版本与批次的一致性。"])
(out / "report.txt").write_text("\n".join(lines) + "\n")
print(json.dumps(dict(out=str(out), points=len(points), fresh_runs=18), ensure_ascii=False))

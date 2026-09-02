# ZVec Flat INT8 contiguous 配置修复与复测

## 结论

此前 `FlatIndexParam(use_contiguous_memory=True)` 只停留在 Collection schema，
Segment 创建实际 Flat 索引时读取了默认返回 `false` 的通用 accessor；INT8 量化副索引的
工厂函数又没有接收这个参数。因此 benchmark 中标为 `contiguous` 的旧结果实际上仍在走
普通 `FlatStreamerEntity`。

本次已修复原始 Flat、INT8 量化 Flat、flush、重开和 compaction 共用的参数链路。两轮
50K × 384D 实测中，真正 contiguous 相对同版本 default 的平均吞吐提升 **9.1%**，平均
延迟降低 **8.2%**，Recall@10 完全相同；代价是 warmup 后 RSS 增加约 **168.6 MiB
(82.3%)**。

## 修复内容

1. `FlatIndexParams::use_flat_contiguous_memory()` 现在返回自身的
   `use_contiguous_memory_`，使 Segment 的通用 Flat materialization 路径不会把直接 Flat
   的设置回落为 `false`。
2. `MakeDefaultQuantVectorIndexParams` 新增 `use_contiguous_memory` 参数。
3. Segment 在创建和加载 INT8 量化副索引时传递该参数。
4. 新增参数工厂和 Segment 创建、flush、重开回归测试。

底层 `FlatStreamer` 的第二道检查保持不变：只有参数为 `true` 且 storage 不是
`MBT_BUFFERPOOL` 时才构造 `FlatContiguousStreamerEntity`。

## 实际路径核验

每轮 benchmark 完成 optimize、close 和只读 reopen 后，直接读取两个 `.proxima` 文件中
的 index meta：

| Layout | `vector.index` | `vector.qindex` |
| --- | --- | --- |
| contiguous round 1 | `use_contiguous_memory:true` | `use_contiguous_memory:true` |
| contiguous round 2 | `use_contiguous_memory:true` | `use_contiguous_memory:true` |
| default round 1 | `use_contiguous_memory:false` | `use_contiguous_memory:false` |
| default round 2 | `use_contiguous_memory:false` | `use_contiguous_memory:false` |

这次的 contiguous/default 对照已经是实际执行路径差异，不再只是 Python 请求参数不同。

## 环境与口径

| 项目 | 配置 |
| --- | --- |
| 平台 | Apple ARM64，macOS 26.6.2 |
| Python | 3.14.6，原生 arm64 |
| ZVec | `0.7.1.dev60`，PR #689 + PR #716 + 本地 NEON 接线 + contiguous 参数修复 |
| 数据 | synthetic，50,000 × 384D，seed 20260901 |
| 查询 | 500 条，warmup 50，repeats 3，topK 10 |
| 距离/量化 | 归一化 FP32 输入 + InnerProduct + INT8 record 量化 |
| Ground truth | NumPy FP32 暴力 InnerProduct topK |

两轮交叉运行顺序：round 1 为 contiguous → default，round 2 为 default → contiguous。

## 原始结果

| Layout | Round | QPS | mean | p50 | p95 | p99 | Recall@10 | warm RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous | 1 | 695.5 | 1.435 ms | 1.417 ms | 1.567 ms | 1.795 ms | 98.32% | 373.3 MiB |
| contiguous | 2 | 720.1 | 1.387 ms | 1.377 ms | 1.469 ms | 1.547 ms | 98.32% | 373.4 MiB |
| default | 1 | 652.7 | 1.529 ms | 1.515 ms | 1.705 ms | 1.813 ms | 98.32% | 204.8 MiB |
| default | 2 | 645.1 | 1.547 ms | 1.517 ms | 1.759 ms | 1.986 ms | 98.32% | 204.8 MiB |

## 两轮均值

| Layout | QPS | mean | p50 | p95 | p99 | Recall@10 | warm RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **contiguous（实际 true）** | **707.8** | **1.411 ms** | **1.397 ms** | **1.518 ms** | **1.671 ms** | 98.32% | 373.4 MiB |
| default | 648.9 | 1.538 ms | 1.516 ms | 1.732 ms | 1.899 ms | 98.32% | 204.8 MiB |
| contiguous 相对变化 | **+9.1%** | **-8.2%** | **-7.8%** | **-12.4%** | **-12.0%** | 0 pp | **+82.3%** |

落盘大小几乎相同（两者约 99.4 MiB）。contiguous 是 reopen 时额外构建内存副本，不是
改变持久化文件格式，所以主要代价体现在 RSS。

## 与旧结果对照

旧的 NEON 接线报告中，所谓 contiguous 的两轮均值为 639.7 QPS / p50 1.508 ms，但其
量化索引 meta 实际为 `false`。修复后真正 contiguous 为 707.8 QPS / p50 1.397 ms，
相对该旧标签吞吐再提升约 **10.6%**、p50 降低约 **7.4%**。

## OpenViking 同轮复测对照

为排除历史结果和本轮机器状态差异，又在相同 Python 进程环境、数据文件和 benchmark
脚本下新鲜复跑 OpenViking 0.4.17.1 两轮。OpenViking 为 1084.1 / 1086.7 QPS，均值
1085.4 QPS，与历史均值 1084.3 QPS 基本一致。

| 指标 | OpenViking 0.4.17.1 | ZVec 0.7.1.dev60 contiguous | 对比 |
| --- | ---: | ---: | ---: |
| QPS | **1085.4** | 707.8 | OpenViking **1.53x** |
| mean | **0.919 ms** | 1.411 ms | ZVec 高 53.5% |
| p50 | **0.889 ms** | 1.397 ms | ZVec 高 57.2% |
| p95 | **1.091 ms** | 1.518 ms | ZVec 高 39.1% |
| p99 | **1.177 ms** | 1.671 ms | ZVec 高 42.0% |
| Recall@10 | 98.12% | **98.32%** | ZVec +0.20 pp |
| 数据写入 | 3.698 s | **1.204 s** | ZVec 快 3.07x |
| 索引构建/持久化 | 1.597 s | **0.087 s** | ZVec 快 18.3x |
| reopen | **0.045 s** | 0.046 s | 基本持平 |
| warm RSS | 701.3 MiB | **373.4 MiB** | OpenViking 为 1.88x |
| ingest 峰值 RSS | 1348.3 MiB | **373.4 MiB** | OpenViking 为 3.61x |
| 落盘 | **97.0 MiB** | 99.4 MiB | OpenViking 少 2.4% |

因此两者不是单向胜负：纯查询吞吐和延迟仍是 OpenViking 明显领先；ZVec 的 Recall 略高，
写入、索引完成时间和内存效率明显更好。`finalize` 阶段两边调用的产品 API 分别是
OpenViking `create_index + close` 和 ZVec `optimize + close`，可用于端到端成本对照，但不应
解读成同一个内部算法的微基准。

## 为什么提升只有约 9%

contiguous 路径消除了普通 Flat 的分块读取和每块临时距离数组，并在一块对齐内存上批量
计算距离，所以延迟和尾延迟稳定下降。但它不会替换 INT8 距离 kernel 或 topK heap。

本 workload 的 INT8 record 为 384 字节向量数据加 20 字节 record tail，即 404 字节；
contiguous entity 按 64 字节对齐把每条 stride 扩到 448 字节，单条扫描字节数增加约
10.9%。对 50K 全扫描而言，收益会被额外内存带宽和更大的 working set 部分抵消。因此它
更像是用内存换约 8%–12% 的查询延迟，而不是新的距离计算加速。

此前微基准中，PR #689 的 NEON INT8 kernel 单次 50K 扫描约 1.115 ms，加简单 topK 后
约 1.134 ms；本次 Collection p50 约 1.397 ms。由此看，剩余差距的主体仍在 INT8 扫描
kernel、record stride 和 topK/search 开销，而不是 contiguous 配置是否生效。

## 验证状态

- `IndexParamsTest.DefaultFlatReferenceParams`：通过。
- 新增 Segment 创建、flush、重开回归：编译通过；当前桌面沙箱启动完整
  `segment_test` 二进制时被系统以 137 终止，未得到 gtest 断言结果。
- Python Collection smoke：创建、optimize、close、reopen、查询通过；原始和量化索引
  均确认落盘为 `true`。
- 正式两轮 Recall@10 均为 98.32%，与 default 完全一致。

原始 JSON：

- `results/zvec_int8_contiguous_contiguous_fix_round1.json`
- `results/zvec_int8_default_contiguous_fix_round1.json`
- `results/zvec_int8_contiguous_contiguous_fix_round2.json`
- `results/zvec_int8_default_contiguous_fix_round2.json`
- `results/openviking_int8_contiguous_fix_compare_round1.json`
- `results/openviking_int8_contiguous_fix_compare_round2.json`

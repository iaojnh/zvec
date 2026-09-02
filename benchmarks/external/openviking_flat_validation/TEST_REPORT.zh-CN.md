# OpenViking Native Local Flat 与 ZVec Flat 性能对比报告

## 测试摘要

本次测试在 Apple ARM64 平台上，对 OpenViking native local Flat 与 ZVec
Flat 的持久化 Collection API 进行同口径对比。测试绕过 embedding、HTTP、文档解析、
过滤和 rerank，只保留向量写入、索引持久化、重新打开和 Top-K 查询。

核心结论：

- 两边均为 Flat 全量扫描；FP32 模式下均达到 100% Recall@10。
- FP32 同布局比较中，OpenViking QPS 比 ZVec 高约 20%，差距较小。
- INT8 模式下，OpenViking QPS 是 ZVec contiguous 的 10.4 倍，p50 延迟低
  10.8 倍；两边 Recall@10 仅相差 0.2 个百分点。
- 根因不是 Flat 算法、Python 包装或记录布局，而是 ZVec 在 Apple ARM 上的 INT8
  Inner Product 距离核进入 float 累加 fallback；OpenViking 使用 INT32 累加并被
  Apple Clang 自动向量化为 NEON。
- 该 INT8 结论具有明显的平台相关性，不能直接外推到 Linux x86 AVX2/AVX512。

## 测试环境

| 项目 | 配置 |
|---|---|
| 机器 | MacBook Pro，Apple M3 Pro |
| CPU | 11 核（5 个性能核 + 6 个能效核），ARM64 |
| 内存 | 36 GB |
| 操作系统 | macOS 26.6.2 |
| Python | 3.14.6 |
| 并发 | 1；每个查询串行计时 |
| 进程隔离 | 每个 backend/mode/layout 单独启动进程 |

## 版本与配置

| 引擎 | 版本 | 配置 |
|---|---|---|
| OpenViking Local | 0.4.17.1（PyPI） | native local backend；Flat；IP；INT8 或显式 FP32 |
| ZVec | 0.7.1.dev16 | checkout `7a682cf87b9202f67dd2deba76f8e695ea6dd2ab`；Flat；IP；INT8 或不量化；分别测试 contiguous/default 布局 |

OpenViking 实际落盘的 INT8 索引元数据为：

```json
{
  "IndexType": "flat",
  "Dimension": 384,
  "Distance": "ip",
  "NormalizeVector": false,
  "Quant": "int8"
}
```

FP32 模式仅将 `Quant` 改为 `float`。输入向量和查询在写入前已经执行 L2
归一化，因此 IP 排序与 cosine 排序等价，同时避免把不同引擎的归一化预处理开销混入
距离核对比。

## 数据与口径

| 项目 | 配置 |
|---|---|
| Base vectors | 50,000 |
| Dimension | 384 |
| Queries | 500 |
| Top-K | 10 |
| Warmup | 50 次 |
| 正式计时 | 每个 query 重复 3 次，共 1,500 次查询调用 |
| Batch size | 1,000 |
| Seed | 20260901 |
| Query noise | 0.03 |

数据由 NumPy 生成 FP32 高斯随机向量并逐行归一化。Query 是随机选取的 base
vector 加入噪声后再次归一化。Ground truth 使用 NumPy FP32 全量 dot-product
计算精确 Top-10。

测试运行两轮；每轮都会重新生成独立 Collection、写入、持久化、关闭并重新打开。
下表使用两轮结果的算术平均值。

## 结果：INT8 Flat

| Backend | 布局 | Recall@10 | QPS | p50 | p95 | p99 | 写入 + finalize | Warm RSS | 落盘大小 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenViking Local | contiguous | 98.12% | **1084.3** | **0.886 ms** | **1.091 ms** | **1.715 ms** | 5.39 s | 702.0 MiB | 97.0 MiB |
| ZVec Flat INT8 | contiguous | **98.32%** | 103.9 | 9.566 ms | 10.402 ms | 11.003 ms | **1.19 s** | **226.2 MiB** | 99.4 MiB |
| ZVec Flat INT8 | default | **98.32%** | 99.4 | 9.910 ms | 11.041 ms | 11.779 ms | 1.21 s | 205.0 MiB | 99.4 MiB |

以 ZVec contiguous 为基准：

- OpenViking QPS 高 `1084.3 / 103.9 = 10.4x`。
- ZVec p50 是 OpenViking 的 `9.566 / 0.886 = 10.8x`。
- ZVec Recall@10 高 0.2 个百分点，差距很小。
- ZVec contiguous 与 default 的 INT8 查询性能仅相差约 4.5%，说明此时不是
  内存布局受限，而是距离计算受限。
- ZVec 写入并 finalize 约快 4.5 倍。

资源指标说明：`Warm RSS` 是查询进程预热后的整进程常驻内存，包含 Python、动态库、
Collection 元数据、缓存及映射页面，不能直接视为纯向量索引内存；落盘大小是完整
Collection 目录大小，不是仅向量文件大小。这两项适合比较本次端到端配置，不宜据此反推
单条向量的理论存储开销。

## 结果：FP32 Flat 对照组

| Backend | 布局 | Recall@10 | QPS | p50 | p95 | p99 | 写入 + finalize | Warm RSS | 落盘大小 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenViking Local | contiguous | 100.00% | **575.1** | **1.667 ms** | **2.199 ms** | **2.865 ms** | 5.69 s | 758.8 MiB | 151.8 MiB |
| ZVec Flat FP32 | contiguous | 100.00% | 478.5 | 2.013 ms | 2.491 ms | 4.369 ms | **1.18 s** | 327.2 MiB | **78.3 MiB** |
| ZVec Flat FP32 | default | 100.00% | 352.0 | 2.783 ms | 3.246 ms | 3.916 ms | **1.15 s** | **253.7 MiB** | **78.3 MiB** |

FP32 是最接近“相同数值实现”的对照组：

- OpenViking QPS 比 ZVec contiguous 高约 20%。
- OpenViking p50 低约 17%。
- ZVec contiguous QPS 比 default 高约 36%，说明距离核高效时，连续内存布局会
  明显影响 Flat 扫描性能。
- 两边 Recall@10 都是 100%。

## INT8 差距根因

### 1. OpenViking 使用 INT32 累加

OpenViking 的 INT8 点积先在 `int32_t` 中完成全部累加，最后只做一次
integer-to-float 转换，再乘以 query/base scale。

安装的 OpenViking 0.4.17.1 二进制中，Apple Clang 将该循环编译为 NEON
`sshll + smlal` widening multiply-accumulate，并使用多个独立 INT32 vector lane
累加，最后水平归约。

### 2. ZVec Apple ARM fallback 使用 float 累加

ZVec 的 `InnerProductInt8Scalar` 复用了通用模板：

```cpp
float sum = 0.0f;
for (size_t i = 0; i < dim; ++i) {
  sum += static_cast<float>(m[i] * q[i]);
}
```

编译器虽然向量化了 INT8 乘法，但为了保持 float 累加顺序，需要反复执行
`scvtf`，并将结果串行 `fadd` 到同一个累加器。生成的长依赖链成为主要瓶颈。

在 384 维下，最坏点积为：

```text
384 × 127 × 127 = 6,193,536
```

该值远低于 INT32 上限，因此本 workload 可以安全地在 INT32 中完成精确累加。

### 3. ZVec IP 未进入 one-to-many 优化路径

ZVec INT8 IP 的 `batch_distance()` 返回
`BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct,...>`，但该 wrapper 只对
Cosine 和 Squared L2 做特殊分派，没有处理 `MinusInnerProduct`，最终落入
`_ComputeBatch()` 并逐条调用单向量距离函数。

单向量 dispatcher 有 AVX2/SSE 实现，但没有 ARM INT8 NEON 实现，因此 Apple
ARM 调用上述 float-accumulator fallback。已有 one-to-many INT8 batch 内核也只对
AVX512 VNNI/AVX2 做了专门分派。

## 距离核微基准

为了把距离核与 Python、Collection、Top-K 和持久化开销分离，额外运行
`kernel_microbench.cpp`。它使用与正式测试相同的 50,000 x 384 工作量，并模拟两边
实际记录步长。Apple Clang `-O3`，四轮中位数如下：

| Kernel | 记录步长 | 每次完整扫描 |
|---|---:|---:|
| OpenViking 式 INT32 累加 | 400 B | **0.388 ms** |
| INT32 累加，ZVec 布局 | 448 B | **0.420 ms** |
| ZVec 式 float 累加 | 448 B | **9.140 ms** |

结论：

- 相同步长下，float 累加比 INT32 累加慢约 22 倍，四轮范围为 19.9x–22.6x。
- 448B 与 400B 步长的中位差距约 7%，记录 padding 不是数量级主因。
- 9.14 ms 的纯 float 距离核与 ZVec 9.566 ms 的完整查询 p50 几乎同量级，说明
  查询耗时主体确实在 INT8 点积循环。
- 完整查询最终只呈现约 10.8x，而不是 22x，是因为 OpenViking 的 0.886 ms 中还
  包含 query 量化、50,000 次间接距离调用、scale 修正、label 读取、Top-K 和 Python
  结果转换；距离核变快后，这些固定成本开始占主导。

## 次要差异

### 量化格式

- OpenViking：对称 per-vector INT8，384B code + 4B scale。
- ZVec：非对称 per-record min/max INT8，384B code + 20B 元数据，包括 scale、
  bias、sum、squared sum 和 integer sum；每条记录还需要执行多项得分修正。

两者不是完全相同的量化公式，因此 INT8 属于产品级同口径对比，不是相同 kernel
实现对比。ZVec 的 Recall 略高可能与非对称量化使用了更完整的动态范围有关。

### 记录布局

- OpenViking 完整记录为 400B：388B encoded vector + 8B label + 4B offset。
- ZVec contiguous 的量化向量为 404B，按 64B 对齐后 stride 为 448B，key 存放在
  独立数组中。

微基准显示该布局差异只贡献约 7%。

### 扫描与 Top-K

OpenViking 直接顺序扫描记录，并仅在候选进入 Top-K 时修改 priority queue。ZVec
contiguous 路径会先构造 vector/key pointer batch，计算距离，再逐条调用 document
heap。距离核修复后，这部分将成为下一阶段优化对象。

## 与 Linux x64 参考结果的关系

用户提供的 16 核 Linux x64 g9i 复测采用 1024D、Cosine、并发 4/8，并报告 ZVec
在裸查询上比 OpenViking 快 5.2x（10K）和 9.0x（100K）。该结果与本报告的 Apple
ARM 方向相反，但不构成矛盾：

- Linux x64 可以进入 ZVec AVX2/AVX512 距离核；Apple ARM 的 INT8 IP 路径没有
  对应 NEON kernel。
- Linux 参考测试使用 Cosine；ZVec Cosine 有独立的 batch/turbo 分派，而本测试使用
  “预归一化 + IP”以直接比较 cosine 等价排序。
- Linux 测试使用 1024D、TopK 20/100 和并发 4/8；本测试使用 384D、TopK 10、
  并发 1。
- 高并发大数据 Flat 扫描还会受到共享内存带宽影响。

因此合理结论应是“当前 ZVec INT8 Flat 性能高度依赖 metric 与 CPU ISA”，而不是将
任一平台的倍率直接外推到所有环境。

## 结论与建议

- OpenViking native local 纯 dense 模式确实是 Flat 暴力扫描；本次实际落盘元数据也
  验证为 `IndexType=flat`。
- 在 Apple M3 Pro、normalized IP、50K x 384D、TopK 10 条件下，OpenViking
  INT8 QPS 是 ZVec contiguous 的 10.4x，主要由 ZVec Apple ARM INT8 IP
  fallback 导致。
- FP32 下两边只差约 20%，说明 Python/Collection API 不是 INT8 10x 差距的主要来源。
- ZVec 优先优化项应为：
  1. 增加 ARM NEON INT8 IP kernel，使用 INT32 lane 累加；
  2. 将 `MinusInnerProduct` 接入 one-to-many batch 路径；
  3. 增加 ARM batch 12/32 专用实现；
  4. 再优化 404B→448B padding、pointer batch 和 heap 构造。
- 完成优化后，应在 Apple ARM、Linux AVX2 和 Linux AVX512 VNNI 上分别复测，不应
  混用跨平台倍率。

## 复现方法

正式测试：

```bash
cd benchmarks/external/openviking_flat_validation

.workspace/venv/bin/python benchmark.py matrix \
  --n 50000 \
  --dim 384 \
  --queries 500 \
  --topk 10 \
  --seed 20260901 \
  --query-noise 0.03 \
  --batch-size 1000 \
  --warmup 50 \
  --repeats 3 \
  --zvec-layouts contiguous,default \
  --results-dir results
```

第二轮将 `--results-dir` 改为 `results/repeat2`，每个 case 会自动删除并重新创建自己的
Collection。

距离核微基准：

```bash
clang++ -O3 -std=c++17 kernel_microbench.cpp \
  -o .workspace/kernel_microbench
.workspace/kernel_microbench
```

## 测试产物

- `benchmark.py`：完整 Collection API benchmark。
- `kernel_microbench.cpp`：INT8 累加与记录步长微基准。
- `results/summary.md`：第一轮结果。
- `results/repeat2/summary.md`：第二轮结果。
- `RESULTS.md`：两轮均值摘要。
- `ANALYSIS.md`：英文根因分析。

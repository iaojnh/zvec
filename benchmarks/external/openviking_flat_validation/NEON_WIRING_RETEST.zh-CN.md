# ZVec Collection INT8 NEON 接线与复测报告

## 结论

在 PR #689 + PR #716 的本地分支上，将 `QuantizedIntegerMetric` 的 INT8
InnerProduct/NormalizedCosine 距离计算接到 Turbo `kAuto` 分发后，Python
`zvec.Collection` 的 50K × 384D Flat INT8 查询已实际命中 ARM NEON kernel。

同机新鲜基线与修改后两轮均值相比：

| Layout | 修改前 | 修改后（两轮均值） | QPS 提升 | p50 降幅 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| contiguous | 95.0 QPS / 10.117 ms | **639.7 QPS / 1.508 ms** | **6.73x** | **85.1%** | 98.32% → 98.32% |
| default | 91.8 QPS / 10.675 ms | **548.1 QPS / 1.794 ms** | **5.97x** | **83.2%** | 98.32% → 98.32% |

Recall 完全不变，说明本次只是替换同一 INT8 record 编码上的距离计算内核，没有改变
量化格式或排序语义。

## 环境与口径

| 项目 | 配置 |
| --- | --- |
| 平台 | Apple ARM64，macOS 26.6.2 |
| Python | 3.14.6，原生 arm64 |
| ZVec | `0.7.1.dev60`，PR #689 + PR #716，本地未提交接线改动 |
| 数据 | synthetic，50,000 × 384D，seed 20260901 |
| 查询 | 500 条，warmup 50，内部 repeats 3，topK 10 |
| 距离/量化 | 归一化 FP32 输入 + InnerProduct + INT8 record 量化 |
| Ground truth | NumPy FP32 暴力 InnerProduct topK |

## 修改内容

`src/core/metric/quantized_integer_metric.cc`：

1. INT8 `InnerProduct` 的单条与 batch 距离改为调用 Turbo record INT8
   `kInnerProduct + kAuto`，不可用时保留原实现 fallback。
2. INT8 `NormalizedCosine` 映射到同一个 InnerProduct Turbo kernel；该路径输入已经归一化，
   与原 `MinusInnerProduct` 语义一致。
3. SquaredEuclidean/Cosine 的硬编码 `kAVX512VNNI` 改为 `kAuto`，让 ARM 选择
   NEON、x86 根据 CPU 选择 VNNI/AVX512/AVX2。
4. `distance_matrix(m, n)` 仅在 `m == 1 && n == 1` 时返回单对 Turbo kernel，
   其他矩阵形状继续使用原矩阵实现。

现有 Collection 的 `Int8StreamingConverter` 和新 Turbo kernel 都使用相同的
`RecordQuantizer` 布局：`[dim bytes INT8][20-byte record tail]`。因此无需重建索引。

## 修改后原始结果

| Layout | Round | QPS | p50 | p95 | p99 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous | 1 | 606.7 | 1.572 ms | 2.118 ms | 2.334 ms | 98.32% |
| contiguous | 2 | 672.7 | 1.445 ms | 1.629 ms | 2.110 ms | 98.32% |
| default | 1 | 497.1 | 1.946 ms | 2.354 ms | 2.779 ms | 98.32% |
| default | 2 | 599.0 | 1.642 ms | 1.820 ms | 2.011 ms | 98.32% |

## 与 OpenViking 历史结果对照

此前相同机器和 workload 下，OpenViking Local INT8 contiguous 的两轮均值为
1084.3 QPS / 0.886 ms。ZVec 修复前为约 97.6 QPS / 10.069 ms，OpenViking 吞吐约
11.1 倍；修复后 ZVec 为 639.7 QPS / 1.508 ms，差距收窄到约 **1.69 倍**。

因此，此前约一个数量级的差距主要来自 ZVec Collection 没有把 PR #689 的 NEON
record INT8/IP kernel 接到实际查询路径，而不是 Flat 或 INT8 方案本身的固有差距。

## 验证

- `quantized_integer_metric_test`：20/20 通过。
- `turbo_neon_distance_test`：5/5 通过；新增断言确认 ARM64 上 `kAuto` 返回的
  record INT8/IP 单条与 batch 函数指针就是 NEON 实现。
- `flat_streamer_test`：19 通过、1 跳过。
- Python Collection 端到端两轮：Recall@10 均为 98.32%。

## 边界

本次倍率只代表 Apple ARM64。Linux x64 g9i 不支持 NEON；同一改动会通过 `kAuto`
选择该机器可用的 AVX2/AVX512 内核，需要在 g9i 上单独构建和复测，不能直接套用本次倍率。

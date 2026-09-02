# ZVec PR #689 Apple ARM 复测报告

## 结论

PR #689 的 NEON 距离核已经成功编译、注册，PR 自带的 ARM 测试也全部通过；但在本次
ZVec Flat + INT8 + Inner Product Collection 查询中，**没有观察到性能提升**。

两轮均值显示，INT8 contiguous 为 99.3 QPS / 9.864 ms p50，与 main 基线的
103.9 QPS / 9.566 ms 基本同一水平。Recall@10 保持 98.32%。当前结果不是新 kernel
算得慢，而是 Flat Collection 的 INT8 IP 查询路径仍未调用 PR 新增的 Turbo NEON
kernel。

## 分支与构建

| 项目 | 值 |
|---|---|
| PR | `alibaba/zvec#689`，`refactor(turbo): migrate neon distance kernels` |
| 本地分支 | `codex/pr-689-neon-benchmark` |
| PR head | `db96b380674b30c3447adb3c95cbcb23b09ed75d` |
| PR base | `upstream/main`：`3d479a18e16c44d95dad70ccfa6992930f407495` |
| 安装版本 | `zvec 0.7.1.dev13` |
| 构建方式 | 独立的 `build-pr689` Release 构建目录，重新生成并安装 Python wheel |
| 测试平台 | Apple M3 Pro，ARM64，macOS 26.6.2，Python 3.14.6 |

PR head 是从 GitHub Pull Request ref 直接获取的，不依赖贡献者 fork 中分支名是否变化。
构建使用全新目录，因此没有复用此前 main 版本的 C++ object。

## PR 自带测试

构建并运行 `turbo_neon_distance_test`：

```text
[  PASSED  ] 4 tests.
```

通过项：

- FP32 NEON 与 scalar 在不同 remainder 维度下结果一致；
- FP16 NEON 与 scalar 在不同 remainder 维度下结果一致；
- record INT8/INT4 NEON 与 scalar 结果一致；
- Turbo dispatcher 能返回新增的 NEON kernels。

这说明 PR 新增实现本身已经进入构建，并且 Turbo 注册表可见。

## 测试口径

与此前 main 版本完全一致，只重新构建 ZVec Collection：

| 项目 | 配置 |
|---|---|
| Base vectors | 50,000 |
| Dimension | 384 |
| Queries | 500 |
| Top-K | 10 |
| Metric | 预归一化向量 + Inner Product；排序等价于 cosine |
| Quantization | INT8；另设 FP32 对照组 |
| Layout | contiguous、default |
| Warmup | 50 次 |
| 正式计时 | 每个 query 重复 3 次，共 1,500 次调用 |
| 并发 | 1 |
| 数据与 ground truth | 复用 main 测试的同一份 `.npy` 文件 |
| 重复轮数 | 2；每轮、每个 case 都重建独立 Collection |

## PR #689 两轮原始结果

| Mode | Layout | Round | Recall@10 | QPS | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | contiguous | 1 | 98.32% | 99.6 | 9.853 ms | 10.787 ms | 11.868 ms |
| INT8 | contiguous | 2 | 98.32% | 99.0 | 9.875 ms | 10.880 ms | 10.971 ms |
| INT8 | default | 1 | 98.32% | 94.7 | 10.370 ms | 11.489 ms | 13.798 ms |
| INT8 | default | 2 | 98.32% | 95.9 | 10.210 ms | 11.269 ms | 11.500 ms |
| FP32 | contiguous | 1 | 100.00% | 510.5 | 1.927 ms | 2.218 ms | 2.397 ms |
| FP32 | contiguous | 2 | 100.00% | 474.0 | 2.107 ms | 2.394 ms | 2.489 ms |
| FP32 | default | 1 | 100.00% | 314.1 | 3.165 ms | 3.589 ms | 4.107 ms |
| FP32 | default | 2 | 100.00% | 319.0 | 3.137 ms | 3.480 ms | 3.656 ms |

## 与 main 两轮均值对比

| Mode | Layout | main QPS | PR #689 QPS | QPS 变化 | main p50 | PR #689 p50 | Recall（main → PR） |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | contiguous | 103.9 | 99.3 | -4.4% | 9.566 ms | 9.864 ms | 98.32% → 98.32% |
| INT8 | default | 99.4 | 95.3 | -4.1% | 9.910 ms | 10.290 ms | 98.32% → 98.32% |
| FP32 | contiguous | 478.5 | 492.2 | +2.9% | 2.013 ms | 2.017 ms | 100.00% → 100.00% |
| FP32 | default | 352.0 | 316.5 | -10.1% | 2.783 ms | 3.151 ms | 100.00% → 100.00% |

这些测试没有在 main 与 PR 之间交错运行，因此几个百分点的变化不宜解释为稳定回归或
提升。可以确定的是：若 PR 的 INT8 NEON kernel 已进入 50K Flat 扫描主路径，应出现远大于
这种波动的改善；当前两轮没有观察到该信号。

沿用此前 OpenViking 两轮均值作为对照：

- INT8 contiguous：OpenViking 1084.3 QPS，PR #689 ZVec 99.3 QPS，前者约
  10.9 倍；p50 为 0.886 ms 对 9.864 ms。
- FP32 contiguous：OpenViking 575.1 QPS，PR #689 ZVec 492.2 QPS，前者约
  1.17 倍。

## 为什么 PR 没有加速这个 workload

### 1. NEON kernel 已注册

PR 在 `src/turbo/turbo.cc` 中为 record-quantized INT8 注册了 NEON 的 Squared L2、
Cosine 和 Inner Product 单向量/批量函数。dispatcher 测试已经证明在 Apple ARM 上能取得
这些函数。

### 2. INT8 IP Collection 查询没有请求 Turbo kernel

`QuantizedIntegerMetric::batch_distance()` 的 `kInnerProduct + DT_INT8` 分支直接返回：

```cpp
BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, int8_t,
                                      12, 2>::ComputeBatch
```

该分支没有调用 `turbo::get_batch_distance_func()`。作为对比，同一函数中的 Squared L2
和 Cosine 分支会先尝试 Turbo。

### 3. 旧 wrapper 仍把 IP 降级为逐向量 fallback

`BaseDistanceBatchWithScoreUnquantized::ComputeBatch()` 只对
`CosineMinusInnerProduct` 和 `SquaredEuclidean` 做特殊 batch 分派；
`MinusInnerProduct` 不在特殊分支中，最终进入 `_ComputeBatch()`，逐条调用旧距离函数。

因此本次 50K Flat + INT8 + IP 的主循环仍是此前分析过的 Apple ARM float-accumulator
fallback，PR 新增的 record INT8 NEON 点积没有被执行。

### 4. 当前 Cosine 分支也存在 ARM 集成限制

当前 `QuantizedIntegerMetric` 的 record INT8 Cosine 路径虽然会查询 Turbo，但请求的 ISA
被显式写为 `kAVX512VNNI`，不是 `kAuto`。在 Apple ARM 上该请求不会选择 NEON row。
所以仅把 NEON row 加入注册表，还不足以让现有 Collection metric 自动使用它。

## 建议的后续改动

要让 PR #689 实际加速 Flat Collection，至少需要补齐 metric 集成：

1. 在 INT8 Inner Product 的 `distance_matrix()` 和 `batch_distance()` 中查询 record
   Turbo kernel；
2. Apple ARM 路径使用 `CpuArchType::kAuto`，让 dispatcher 选择 NEON；
3. 保留原 `BaseDistanceBatch...` 作为无可用 Turbo kernel 时的 fallback；
4. 增加从 Collection/Flat 查询入口到 Turbo NEON kernel 的集成测试，而不仅是直接调用
   dispatcher 的单元测试；
5. 修改后复用本报告同一数据，再做 main、PR、OpenViking 三方交错复测。

## 原始产物

- `results/pr689_round1/`：第一轮四个 ZVec JSON。
- `results/pr689_round2/`：第二轮四个 ZVec JSON。
- `results/`、`results/repeat2/`：此前 main 两轮基线。
- `benchmark.py`：本次 Collection API benchmark。

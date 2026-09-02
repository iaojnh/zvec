# ZVec PR #689 + PR #716 Apple ARM 合并复测报告

## 结论

PR #716 已合并到此前的 PR #689 本地测试分支，并使用全新的 Release 构建重新安装。
相关 NEON 与 Flat 单元测试全部通过，但在 Python Collection API 的 Flat + INT8 +
Inner Product workload 中，**仍未观察到 NEON 加速**。

两轮均值为：

- INT8 contiguous：97.6 QPS / 10.069 ms p50；
- INT8 default：95.2 QPS / 10.261 ms p50；
- Recall@10 均为 98.32%。

与仅合并 PR #689 的结果基本相同。根因是 PR #716 为 Flat builder/searcher/streamer
增加了“调用方显式传入 Turbo quantizer”时的加速路径，但当前 Python Collection 的
INT8 建库与 reopen 流程仍使用旧的 `Int8StreamingConverter + Int8StreamingReformer +
QuantizedInteger`，没有把 Turbo quantizer 传给 Flat。

## 分支与构建

| 项目 | 值 |
|---|---|
| 本地分支 | `codex/pr-689-neon-benchmark` |
| PR #689 head | `db96b380674b30c3447adb3c95cbcb23b09ed75d` |
| PR #716 head | `34835739250b2b23264a9b3ad7d516745eabb156` |
| 合并提交 | `6f770a69dc966f51ca807fe2e2c80d4785029e74` |
| 安装版本 | `zvec 0.7.1.dev60` |
| 构建目录 | `build-pr689-pr716`，Release，全新 ZVec object |
| 平台 | Apple M3 Pro，ARM64，macOS 26.6.2，Python 3.14.6 |

首次构建时 GitHub/Apache 的 Boost、Thrift 下载超时。最终构建复用了此前构建中版本、
URL 与 SHA256 完全相同且已校验的 Arrow 第三方源码缓存；没有复用旧 ZVec object。

## 合并后单元测试

| 测试 | 结果 |
|---|---:|
| `turbo_neon_distance_test` | 4/4 通过 |
| `flat_builder_test` | 15/15 通过 |
| `flat_streamer_test` | 19 通过，1 个预期跳过 |

测试证明：

- PR #689 的 NEON FP32/FP16/record INT8/INT4 kernel 正确且 dispatcher 可见；
- PR #716 的 Flat builder/streamer 显式 Turbo quantizer 路径工作正常；
- 问题位于上层 Collection 到该新路径的接线，而不是新增 kernel 或 Flat overload 本身。

## 测试口径

与 main、仅 PR #689 两次测试完全相同：

| 项目 | 配置 |
|---|---|
| Base vectors | 50,000 |
| Dimension | 384 |
| Queries | 500 |
| Top-K | 10 |
| Metric | 输入预归一化 + Inner Product；排序等价于 cosine |
| Quantization | INT8；另设 FP32 对照组 |
| Layout | contiguous、default |
| Warmup | 50 次 |
| 正式计时 | 每 query 重复 3 次，共 1,500 次调用 |
| 并发 | 1 |
| 数据与 ground truth | 复用相同 `.npy` 文件 |
| 重复轮数 | 2；每轮每个 case 重建独立 Collection |

## 合并版本两轮原始结果

| Mode | Layout | Round | Recall@10 | QPS | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | contiguous | 1 | 98.32% | 97.0 | 10.097 ms | 11.056 ms | 13.389 ms |
| INT8 | contiguous | 2 | 98.32% | 98.2 | 10.041 ms | 10.923 ms | 11.310 ms |
| INT8 | default | 1 | 98.32% | 95.3 | 10.219 ms | 11.268 ms | 13.073 ms |
| INT8 | default | 2 | 98.32% | 95.1 | 10.304 ms | 11.239 ms | 11.868 ms |
| FP32 | contiguous | 1 | 100.00% | 482.2 | 2.041 ms | 2.374 ms | 2.494 ms |
| FP32 | contiguous | 2 | 100.00% | 474.0 | 2.101 ms | 2.400 ms | 2.564 ms |
| FP32 | default | 1 | 100.00% | 323.6 | 3.056 ms | 3.500 ms | 3.723 ms |
| FP32 | default | 2 | 100.00% | 310.8 | 3.199 ms | 3.687 ms | 4.150 ms |

## 三版本两轮均值对比

| Mode | Layout | main QPS / p50 | 仅 PR #689 QPS / p50 | PR #689 + #716 QPS / p50 | Recall |
|---|---|---:|---:|---:|---:|
| INT8 | contiguous | 103.9 / 9.566 ms | 99.3 / 9.864 ms | **97.6 / 10.069 ms** | 98.32% |
| INT8 | default | 99.4 / 9.910 ms | 95.3 / 10.290 ms | **95.2 / 10.261 ms** | 98.32% |
| FP32 | contiguous | 478.5 / 2.013 ms | 492.2 / 2.017 ms | **478.1 / 2.071 ms** | 100.00% |
| FP32 | default | 352.0 / 2.783 ms | 316.5 / 3.151 ms | **317.2 / 3.127 ms** | 100.00% |

合并版本相对 main：

- INT8 contiguous QPS -6.0%，p50 +5.3%；
- INT8 default QPS -4.2%，p50 +3.5%；
- FP32 contiguous QPS -0.1%，基本不变。

测试没有按 main/PR 交错运行，几个百分点不应解释为稳定性能回归。可以确定的是，预期的
NEON 数量级提升没有出现，INT8 仍处在约 10 ms 的旧距离路径。

沿用此前 OpenViking 均值，INT8 contiguous 为 1084.3 QPS / 0.886 ms；相比本次合并后
ZVec 的 97.6 QPS / 10.069 ms，吞吐约为 11.1 倍，p50 约低 11.4 倍。

## 为什么 PR #716 仍未加速 Collection

### PR #716 实际提供了什么

PR #716 为 Flat builder、searcher 和 streamer 增加接收
`std::shared_ptr<turbo::Quantizer>` 的 overload。只要调用方传入 quantizer：

- query 会由 Turbo quantizer 编码；
- contiguous Flat 扫描会调用 `calc_distance_dp_query_batch()`；
- PR #689 注册的 Apple ARM NEON record INT8 kernel 可由 `kAuto` dispatcher 选中。

相关直接单元测试已经通过。

### Python Collection 当前实际落盘路径

本次 INT8 Collection 的 `vector.qindex.2.proxima` 元数据为：

```json
{
  "metric": {"name": "QuantizedInteger"},
  "reformer": {"name": "Int8StreamingReformer"},
  "streamer": {
    "name": "FlatStreamer",
    "params": {"proxima.flat.use_contiguous_memory": false}
  },
  "converter": {"name": "Int8StreamingConverter"}
}
```

其中没有 Turbo quantizer 名称。Collection 的量化副索引继续通过 core-interface
`FlatIndex::CreateAndInitConverterReformer()` 创建旧 converter/reformer；
`FlatIndex::CreateAndInitStreamer()` 仍调用不带 quantizer 的 `streamer_->init(meta,
params)`。

因此 PR #716 新增的 Flat overload 没有被 Python Collection 使用。`use_contiguous_memory`
在量化副索引上也未从用户 Flat 参数传递，解释了 INT8 contiguous/default 性能一直非常
接近。

## 后续建议

若目标是加速正式 Collection API，需要继续补齐：

1. 让 Collection/segment 的量化 Flat 索引创建并持久化 Turbo `Int8Quantizer` 元数据；
2. reopen 时由 core-interface 或 IndexFlow 创建相同 quantizer，并传入 Flat streamer/
   searcher；
3. 将用户的 `use_contiguous_memory` 传递到量化副索引，而不是固定为 false；
4. 增加 Python Collection 端到端集成测试，断言 query 确实命中 NEON batch kernel；
5. 完成后继续复用本报告数据做三版本交错复测。

## 原始产物

- `results/pr689_pr716_round1/`：合并版本第一轮 JSON；
- `results/pr689_pr716_round2/`：合并版本第二轮 JSON；
- `results/pr689_round1/`、`results/pr689_round2/`：仅 PR #689；
- `results/`、`results/repeat2/`：main 基线；
- `benchmark.py`：相同 Collection API benchmark。

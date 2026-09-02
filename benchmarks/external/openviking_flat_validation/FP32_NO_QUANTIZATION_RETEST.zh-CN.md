# OpenViking 与 ZVec Flat 并发测试

## 结论

并发数会改变两边的性能排序：

- 并发 1：OpenViking 更快。
- 并发 4：INT8 仍是 OpenViking 更快；FP32 变成 ZVec 略快。
- 并发 8：ZVec 在 INT8 和 FP32 下都更快。

最明显的是 INT8：OpenViking 从并发 4 提升到并发 8 后吞吐下降，而 ZVec 继续增长，
最终由 ZVec 领先 **27%**。

## 测试配置

| 项目 | 配置 |
|---|---|
| 机器 | Apple M3 Pro，11 核（5P+6E），36 GB |
| 数据 | 50,000 × 384D，500 条 query |
| 查询 | TopK 10；并发 1、4、8 |
| 索引 | Flat contiguous；分别测试 INT8 和 FP32 |
| 距离 | Inner Product |
| OpenViking | 0.4.17.1，native local |
| ZVec | 0.7.1.dev60，NEON 与 contiguous 路径已生效 |

两边使用完全相同的数据、查询和并发模型。每个 case 执行 500×3 次查询；并发 4/8
各交叉运行两轮，表中为两轮平均值。

## INT8 结果

| 并发 | OpenViking QPS / P50 | ZVec QPS / P50 | 更快的一方 |
|---:|---:|---:|---:|
| 1 | **1085 / 0.889 ms** | 708 / 1.397 ms | OpenViking **1.53x** |
| 4 | **3025 / 1.149 ms** | 2226 / 1.653 ms | OpenViking **1.36x** |
| 8 | 2627 / 2.874 ms | **3334 / 2.136 ms** | ZVec **1.27x** |

Recall@10：OpenViking 98.12%，ZVec 98.32%，并发变化不影响结果正确性。

## FP32 无量化结果

| 并发 | OpenViking QPS / P50 | ZVec QPS / P50 | 更快的一方 |
|---:|---:|---:|---:|
| 1 | **590 / 1.633 ms** | 522 / 1.877 ms | OpenViking **1.13x** |
| 4 | 1171 / 3.255 ms | **1260 / 2.907 ms** | ZVec **1.08x** |
| 8 | 1217 / 6.360 ms | **1369 / 5.729 ms** | ZVec **1.12x** |

两边 Recall@10 均为 100%。

## 并发扩展能力

以下倍率均相对于各自的并发 1：

| 模式 | Backend | 并发 4 | 并发 8 |
|---|---|---:|---:|
| INT8 | OpenViking | 2.79x | 2.42x |
| INT8 | ZVec | **3.14x** | **4.71x** |
| FP32 | OpenViking | 1.98x | 2.06x |
| FP32 | ZVec | **2.41x** | **2.62x** |

ZVec 的多线程扩展明显更好。OpenViking INT8 在并发 4 达到峰值，并发 8 时出现资源竞争；
FP32 两边较早受内存带宽限制，因此扩展倍率都不高。

## 最终判断

之前“OpenViking INT8 快 1.53x、FP32 快 1.13x”的结论只适用于单并发。

在更接近 OpenViking 官方 benchmark 的多并发条件下：

- 并发 4：双方互有胜负。
- 并发 8：ZVec 全面领先，INT8 领先 27%，FP32 领先 12%。

这也解释了为什么其他报告中的 ZVec 聚合 QPS 看起来更高：ZVec 能更有效地利用多个并发
查询，而 OpenViking 的优势主要集中在单查询延迟。

原始结果：`results/concurrency_50k_*_round*.json`。测试程序：`benchmark.py`。

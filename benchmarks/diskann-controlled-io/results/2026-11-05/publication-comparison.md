# DiskANN 全平台真实 I/O 限速实测

统一使用 100,000 条、128 维 FP32 SIFT 数据及同一份已训练索引，固定查询参数。单并发 C++ 核心接口；每个索引预热 1,000 条查询后，计时执行 1,000 条查询 × 3 遍，3 个独立进程试验取中位数。

P95 按真实搜索调用耗时统计，QPS 按完整计时区间的实际经过时间计算；原生文件读取与限速等待均计入。HNSW 预热后驻留内存，DiskANN 保留生产文件读取路径，不预载完整图和向量。

RSS 为查询进程内存，Windows 为工作集，单位 MB（十进制）；不包含未映射的系统文件缓存。CPU、内存和存储仍为各 CI runner 实际配置；受控 IOPS 是已验证的速率上限。移动端均为模拟器，结果不代表手机真机。

来源：https://github.com/iaojnh/zvec/actions/runs/41991413997；提交 `43a2af9d16b10b258a45a03223136959c16f7466`。每个探针档位先预热 8,192 次，再计时读取 8,192 次（每次 4 KiB）。IOPS 限制针对原生文件读请求，物理设备操作可能被固件或虚拟化层合并或放大。

下表所有数值按 **HNSW → DiskANN** 排列。

## 2,000 IOPS 上限

最多 8 个请求并发；该档位须先通过真实速率、半速对照和缓存验证。

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 102.44 → 24.01 | 97.28% → 98.67% | 0.086 → 28.905 | 15,224.25 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| linux-x86_64 | 103.86 → 24.72 | 97.28% → 98.67% | 0.046 → 28.781 | 27,702.93 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| macos-arm64 | 99.89 → 19.35 | 97.28% → 98.67% | 0.206 → 30.307 | 11,166.28 → 38.13 | Apple per-range eviction / 1,957 |
| macos-x86_64 | 96.92 → 16.66 | 97.28% → 98.67% | 0.151 → 29.659 | 10,486.14 → 37.27 | Apple per-range eviction / 1,913 |
| windows-arm64 | 95.10 → 14.40 | 97.28% → 98.67% | 0.086 → 29.058 | 14,894.55 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |
| windows-x86_64 | 93.44 → 17.06 | 97.28% → 98.67% | 0.076 → 29.155 | 17,736.59 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |

移动端模拟器：

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 95.15 → 15.56 | 97.28% → 98.67% | 0.096 → 29.268 | 13,005.90 → 38.95 | Android O_DIRECT requested / 2,000 |
| ios-arm64-simulator **模拟器** | 132.01 → 37.67 | 97.28% → 98.67% | 0.248 → 30.567 | 8,604.28 → 38.52 | Apple per-range eviction / 1,977 |

## 10,000 IOPS 上限

最多 8 个请求并发；该档位须先通过真实速率、半速对照和缓存验证。

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 102.45 → 24.00 | 97.28% → 98.67% | 0.095 → 5.856 | 13,742.65 → 193.96 | Linux native reader; page-cache bypass checked by OS read bytes / 9,956 |
| linux-x86_64 | — | — | — | — | 未通过此档位验证，留空 |
| macos-arm64 | — | — | — | — | 未通过此档位验证，留空 |
| macos-x86_64 | — | — | — | — | 未通过此档位验证，留空 |
| windows-arm64 | — | — | — | — | 未通过此档位验证，留空 |
| windows-x86_64 | — | — | — | — | 未通过此档位验证，留空 |

移动端模拟器：

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | — | — | — | — | 未通过此档位验证，留空 |
| ios-arm64-simulator **模拟器** | — | — | — | — | 未通过此档位验证，留空 |

## 20,000 IOPS 上限

最多 8 个请求并发；该档位须先通过真实速率、半速对照和缓存验证。

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 102.45 → 24.01 | 97.28% → 98.67% | 0.092 → 3.073 | 14,333.73 → 375.82 | Linux native reader; page-cache bypass checked by OS read bytes / 19,291 |
| linux-x86_64 | — | — | — | — | 未通过此档位验证，留空 |
| macos-arm64 | — | — | — | — | 未通过此档位验证，留空 |
| macos-x86_64 | — | — | — | — | 未通过此档位验证，留空 |
| windows-arm64 | — | — | — | — | 未通过此档位验证，留空 |
| windows-x86_64 | — | — | — | — | 未通过此档位验证，留空 |

移动端模拟器：

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | — | — | — | — | 未通过此档位验证，留空 |
| ios-arm64-simulator **模拟器** | — | — | — | — | 未通过此档位验证，留空 |

## 原生不限速参考

各平台原生磁盘不同，本组展示各自 CI 环境下的性能，不作为操作系统排名。

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 102.45 → 24.01 | 97.28% → 98.67% | 0.085 → 12.284 | 15,405.11 → 381.14 | Linux native reader; page-cache bypass checked by OS read bytes / 19,564 |
| linux-x86_64 | 103.69 → 24.73 | 97.28% → 98.67% | 0.047 → 3.352 | 26,923.58 → 339.44 | Linux native reader; page-cache bypass checked by OS read bytes / 17,424 |
| macos-arm64 | 99.76 → 19.07 | 97.28% → 98.67% | 0.226 → 6.229 | 10,206.07 → 252.08 | Apple per-range eviction / 12,939 |
| macos-x86_64 | 96.89 → 16.67 | 97.28% → 98.67% | 0.145 → 20.565 | 9,825.02 → 58.76 | Apple per-range eviction / 3,016 |
| windows-arm64 | 95.11 → 18.62 | 97.28% → 98.67% | 0.087 → 5.057 | 14,818.25 → 238.67 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 12,251 |
| windows-x86_64 | 93.47 → 17.07 | 97.28% → 98.67% | 0.072 → 5.313 | 18,345.07 → 221.42 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 11,365 |

移动端模拟器：

| 环境 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存条件 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 95.21 → 15.64 | 97.28% → 98.67% | 0.086 → 8.190 | 16,880.48 → 143.91 | Android O_DIRECT requested / 7,387 |
| ios-arm64-simulator **模拟器** | 132.10 → 50.68 | 97.28% → 98.67% | 0.263 → 11.903 | 8,396.37 → 154.62 | Apple per-range eviction / 7,937 |

## 缓存与验证范围

- Linux/Android：结合预热缓冲读取对照与 OS 存储读字节验证。Android 使用 CI 模拟器的未加密控制目录。
- macOS/iOS：生产读取器按请求范围驱逐缓存，并核对 OS 存储读字节；iOS 由宿主观察模拟器进程计数。
- Windows：原生 FILE_FLAG_NO_BUFFERING；进程 I/O 计数为逻辑传输量，未独立验证物理设备缓存。
- 设备及虚拟化宿主层的缓存不能据此全部排除，数据仅对应记录的 CI 环境。

## 结果来源与重跑

按每个平台的首份完整合格产物汇总，不按性能高低挑选。未达标及后续重跑结果全部保存在 `all-attempts/`，选择规则、产物编号与文件校验值见 `selection.json`。

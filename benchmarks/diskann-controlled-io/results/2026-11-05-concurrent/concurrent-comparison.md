# HNSW / DiskANN 八平台并发实测

同一份 100,000 条、128 维 SIFT FP32 数据与已训练索引，固定查询参数。
本轮只新增 2、4、8 个查询工作线程；1 线程直接沿用先前复核通过的结果，未重跑。1 线程来自此前的 CI 实例，不能据此把相对它的差异全部归因于线程数。

每个进程共享一个索引。1,000 条评估查询合计重复 3 遍，共 3,000 次调用，按线程均匀分摊，不随线程数增加总查询量。先预热一次工作负载，再在三个独立进程试验中测量并取中位数。

IOPS、带宽和最多 8 个在途读取由所有线程共享。P95 为所有线程实际搜索调用耗时的分位数，包含限速与 I/O 队列等待；QPS 为总完成查询数除以整个进程计时区间的真实经过时间。

RSS 是所有工作线程及其查询缓冲区仍存活时的进程总量；Windows 使用工作集，单位为十进制 MB。HNSW 已预热驻留内存；DiskANN 使用原生文件读取，缓存条件逐行列明。

新增测试来源：https://github.com/iaojnh/zvec/actions/runs/42003247275；提交 `099cd1c2c14997df581ff1824395388bcd31aeef`。1 线程来源：https://github.com/iaojnh/zvec/actions/runs/41991413997；提交 `43a2af9d16b10b258a45a03223136959c16f7466`。

数值按 **HNSW → DiskANN** 排列。† 表示宿主未通过该档位的吞吐资格；仍列出相同配置上限下的真实结果与实际 IOPS。

## 进程总上限 2,000 IOPS

桌面 CI：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 1（沿用） | 102.44 → 24.01 | 97.28% → 98.67% | 0.086 → 28.905 | 15,224.25 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| linux-arm64 | 2 | 103.00 → 24.59 | 97.28% → 98.67% | 0.085 → 59.962 | 30,613.51 → 38.88 | Linux native reader; page-cache bypass checked by OS read bytes / 1,996 |
| linux-arm64 | 4 | 103.53 → 25.59 | 97.28% → 98.67% | 0.091 → 120.494 | 55,019.98 → 38.93 | Linux native reader; page-cache bypass checked by OS read bytes / 1,998 |
| linux-arm64 | 8 | 104.56 → 27.39 | 97.28% → 98.67% | 0.100 → 240.908 | 51,074.37 → 38.92 | Linux native reader; page-cache bypass checked by OS read bytes / 1,998 |
| linux-x86_64 | 1（沿用） | 103.86 → 24.72 | 97.28% → 98.67% | 0.046 → 28.781 | 27,702.93 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| linux-x86_64 | 2 | 104.27 → 25.31 | 97.28% → 98.67% | 0.049 → 59.561 | 54,798.47 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| linux-x86_64 | 4 | 104.96 → 26.32 | 97.28% → 98.67% | 0.080 → 120.302 | 65,041.97 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| linux-x86_64 | 8 | 105.92 → 28.18 | 97.28% → 98.67% | 0.080 → 240.071 | 65,035.08 → 38.96 | Linux native reader; page-cache bypass checked by OS read bytes / 2,000 |
| macos-arm64 | 1（沿用） | 99.89 → 19.35 | 97.28% → 98.67% | 0.206 → 30.307 | 11,166.28 → 38.13 | Apple per-range eviction / 1,957 |
| macos-arm64 | 2 | 100.20 → 19.53 | 97.28% → 98.67% | 0.209 → 62.865 | 22,606.47 → 37.44 | Apple per-range eviction / 1,922 |
| macos-arm64 | 4 | 100.70 → 20.14 | 97.28% → 98.67% | 0.251 → 129.433 | 24,858.77 → 37.89 | Apple per-range eviction / 1,945 |
| macos-arm64 | 8 | 101.52 → 21.43 | 97.28% → 98.67% | 0.212 → 249.941 | 31,338.38 → 37.91 | Apple per-range eviction / 1,946 |
| macos-x86_64 | 1（沿用） | 96.92 → 16.66 | 97.28% → 98.67% | 0.151 → 29.659 | 10,486.14 → 37.27 | Apple per-range eviction / 1,913 |
| macos-x86_64 | 2 | 97.37 → 16.97 | 97.28% → 98.67% | 0.186 → 64.533 | 13,934.52 → 36.76 | Apple per-range eviction / 1,887 |
| macos-x86_64 | 4 | 97.82 → 17.48 | 97.28% → 98.67% | 0.169 → 130.334 | 31,065.10 → 37.39 | Apple per-range eviction / 1,919 |
| macos-x86_64 | 8 | 98.71 → 18.43 | 97.28% → 98.67% | 0.169 → 247.098 | 29,551.62 → 38.05 | Apple per-range eviction / 1,953 |
| windows-arm64 | 1（沿用） | 95.10 → 14.40 | 97.28% → 98.67% | 0.086 → 29.058 | 14,894.55 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |
| windows-arm64 | 2 | 95.92 → 17.38 | 97.28% → 98.67% | 0.117 → 59.517 | 21,907.85 → 38.95 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |
| windows-arm64 | 4 | 96.85 → 18.21 | 97.28% → 98.67% | 0.119 → 119.941 | 39,399.03 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |
| windows-arm64 | 8 | 97.60 → 19.42 | 97.28% → 98.67% | 0.125 → 240.433 | 39,075.22 → 38.90 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 1,997 |
| windows-x86_64 | 1（沿用） | 93.44 → 17.06 | 97.28% → 98.67% | 0.076 → 29.155 | 17,736.59 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |
| windows-x86_64 | 2 | 94.00 → 17.08 | 97.28% → 98.67% | 0.101 → 59.426 | 29,612.32 → 38.95 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 1,999 |
| windows-x86_64 | 4 | 94.86 → 17.56 | 97.28% → 98.67% | 0.118 → 120.320 | 45,157.40 → 38.94 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 1,999 |
| windows-x86_64 | 8 | 95.69 → 18.50 | 97.28% → 98.67% | 0.118 → 239.853 | 42,965.91 → 38.96 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 2,000 |

移动端模拟器：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 1（沿用） | 95.15 → 15.56 | 97.28% → 98.67% | 0.096 → 29.268 | 13,005.90 → 38.95 | Android O_DIRECT requested / 2,000 |
| android-x86_64-emulator **模拟器** | 2 | 95.08 → 16.05 | 97.28% → 98.67% | 0.110 → 60.219 | 27,032.70 → 38.87 | Android O_DIRECT requested / 1,995 |
| android-x86_64-emulator **模拟器** | 4 | 95.70 → 16.29 | 97.28% → 98.67% | 0.112 → 120.261 | 27,619.36 → 38.90 | Android O_DIRECT requested / 1,997 |
| android-x86_64-emulator **模拟器** | 8 | 96.41 → 17.31 | 97.28% → 98.67% | 0.099 → 240.135 | 29,703.82 → 38.95 | Android O_DIRECT requested / 1,999 |
| ios-arm64-simulator **模拟器** | 1（沿用） | 132.01 → 37.67 | 97.28% → 98.67% | 0.248 → 30.567 | 8,604.28 → 38.52 | Apple per-range eviction / 1,977 |
| ios-arm64-simulator **模拟器** † | 2 | 132.27 → 44.83 | 97.28% → 98.67% | 0.306 → 63.116 | 10,077.94 → 37.99 | Apple per-range eviction / 1,950 |
| ios-arm64-simulator **模拟器** † | 4 | 132.37 → 51.87 | 97.28% → 98.67% | 0.230 → 132.249 | 23,463.83 → 37.27 | Apple per-range eviction / 1,913 |
| ios-arm64-simulator **模拟器** † | 8 | 132.69 → 52.67 | 97.28% → 98.67% | 0.215 → 251.375 | 24,904.03 → 38.00 | Apple per-range eviction / 1,950 |

## 进程总上限 10,000 IOPS

桌面 CI：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 1（沿用） | 102.45 → 24.00 | 97.28% → 98.67% | 0.095 → 5.856 | 13,742.65 → 193.96 | Linux native reader; page-cache bypass checked by OS read bytes / 9,956 |
| linux-arm64 | 2 | 103.01 → 24.59 | 97.28% → 98.67% | 0.096 → 12.020 | 27,035.45 → 193.83 | Linux native reader; page-cache bypass checked by OS read bytes / 9,949 |
| linux-arm64 | 4 | 103.53 → 25.59 | 97.28% → 98.67% | 0.100 → 24.088 | 50,820.94 → 194.80 | Linux native reader; page-cache bypass checked by OS read bytes / 9,999 |
| linux-arm64 | 8 | 104.57 → 27.39 | 97.28% → 98.67% | 0.096 → 48.380 | 53,697.80 → 194.22 | Linux native reader; page-cache bypass checked by OS read bytes / 9,969 |
| linux-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| linux-x86_64 | 2 | 104.44 → 25.36 | 97.28% → 98.67% | 0.048 → 11.987 | 54,802.74 → 194.81 | Linux native reader; page-cache bypass checked by OS read bytes / 10,000 |
| linux-x86_64 | 4 | 104.82 → 26.47 | 97.28% → 98.67% | 0.081 → 24.151 | 62,470.97 → 194.82 | Linux native reader; page-cache bypass checked by OS read bytes / 10,000 |
| linux-x86_64 | 8 | 105.97 → 28.06 | 97.28% → 98.67% | 0.081 → 48.159 | 63,790.12 → 194.82 | Linux native reader; page-cache bypass checked by OS read bytes / 10,000 |
| macos-arm64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| macos-arm64 | 2 | 100.22 → 19.51 | 97.28% → 98.67% | 0.192 → 17.370 | 25,041.14 → 161.86 | Apple per-range eviction / 8,308 |
| macos-arm64 | 4 | 100.45 → 20.22 | 97.28% → 98.67% | 0.213 → 32.986 | 32,453.42 → 161.92 | Apple per-range eviction / 8,311 |
| macos-arm64 | 8 | 100.96 → 21.51 | 97.28% → 98.67% | 0.271 → 78.196 | 21,394.85 → 149.61 | Apple per-range eviction / 7,680 |
| macos-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |

移动端模拟器：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 8 | — | — | — | — | 未完成或该可选档位未达标 |

## 进程总上限 20,000 IOPS

桌面 CI：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 1（沿用） | 102.45 → 24.01 | 97.28% → 98.67% | 0.092 → 3.073 | 14,333.73 → 375.82 | Linux native reader; page-cache bypass checked by OS read bytes / 19,291 |
| linux-arm64 | 2 | 102.99 → 24.59 | 97.28% → 98.67% | 0.097 → 6.259 | 26,935.38 → 381.72 | Linux native reader; page-cache bypass checked by OS read bytes / 19,594 |
| linux-arm64 | 4 | 103.53 → 25.58 | 97.28% → 98.67% | 0.095 → 12.293 | 52,367.84 → 383.16 | Linux native reader; page-cache bypass checked by OS read bytes / 19,667 |
| linux-arm64 | 8 | 104.56 → 27.39 | 97.28% → 98.67% | 0.094 → 24.857 | 52,829.23 → 378.93 | Linux native reader; page-cache bypass checked by OS read bytes / 19,451 |
| linux-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| linux-x86_64 | 2 | 104.40 → 25.39 | 97.28% → 98.67% | 0.049 → 6.101 | 54,692.10 → 386.71 | Linux native reader; page-cache bypass checked by OS read bytes / 19,850 |
| linux-x86_64 | 4 | 104.60 → 26.43 | 97.28% → 98.67% | 0.081 → 12.185 | 62,973.32 → 388.26 | Linux native reader; page-cache bypass checked by OS read bytes / 19,930 |
| linux-x86_64 | 8 | 105.82 → 28.16 | 97.28% → 98.67% | 0.082 → 24.176 | 63,281.99 → 389.19 | Linux native reader; page-cache bypass checked by OS read bytes / 19,977 |
| macos-arm64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| macos-arm64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-arm64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-arm64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| macos-x86_64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-arm64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| windows-x86_64 | 8 | — | — | — | — | 未完成或该可选档位未达标 |

移动端模拟器：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| android-x86_64-emulator **模拟器** | 8 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 1（沿用） | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 2 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 4 | — | — | — | — | 未完成或该可选档位未达标 |
| ios-arm64-simulator **模拟器** | 8 | — | — | — | — | 未完成或该可选档位未达标 |

## 原生不限速参考（进程总读取深度仍为 8）

桌面 CI：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| linux-arm64 | 1（沿用） | 102.45 → 24.01 | 97.28% → 98.67% | 0.085 → 12.284 | 15,405.11 → 381.14 | Linux native reader; page-cache bypass checked by OS read bytes / 19,564 |
| linux-arm64 | 2 | 103.00 → 24.59 | 97.28% → 98.67% | 0.094 → 19.861 | 27,409.09 → 381.64 | Linux native reader; page-cache bypass checked by OS read bytes / 19,590 |
| linux-arm64 | 4 | 103.51 → 25.58 | 97.28% → 98.67% | 0.084 → 25.568 | 57,337.26 → 381.24 | Linux native reader; page-cache bypass checked by OS read bytes / 19,569 |
| linux-arm64 | 8 | 104.56 → 27.39 | 97.28% → 98.67% | 0.095 → 39.691 | 52,327.88 → 381.67 | Linux native reader; page-cache bypass checked by OS read bytes / 19,591 |
| linux-x86_64 | 1（沿用） | 103.69 → 24.73 | 97.28% → 98.67% | 0.047 → 3.352 | 26,923.58 → 339.44 | Linux native reader; page-cache bypass checked by OS read bytes / 17,424 |
| linux-x86_64 | 2 | 104.42 → 25.31 | 97.28% → 98.67% | 0.049 → 2.872 | 54,902.89 → 890.59 | Linux native reader; page-cache bypass checked by OS read bytes / 45,714 |
| linux-x86_64 | 4 | 104.74 → 26.33 | 97.28% → 98.67% | 0.080 → 5.209 | 64,960.72 → 914.03 | Linux native reader; page-cache bypass checked by OS read bytes / 46,917 |
| linux-x86_64 | 8 | 105.85 → 28.20 | 97.28% → 98.67% | 0.081 → 11.748 | 64,619.12 → 803.34 | Linux native reader; page-cache bypass checked by OS read bytes / 41,235 |
| macos-arm64 | 1（沿用） | 99.76 → 19.07 | 97.28% → 98.67% | 0.226 → 6.229 | 10,206.07 → 252.08 | Apple per-range eviction / 12,939 |
| macos-arm64 | 2 | 99.98 → 19.53 | 97.28% → 98.67% | 0.212 → 18.391 | 20,158.51 → 158.29 | Apple per-range eviction / 8,125 |
| macos-arm64 | 4 | 100.47 → 20.19 | 97.28% → 98.67% | 0.212 → 37.499 | 31,407.68 → 157.28 | Apple per-range eviction / 8,073 |
| macos-arm64 | 8 | 100.97 → 21.63 | 97.28% → 98.67% | 0.204 → 71.273 | 35,448.51 → 161.03 | Apple per-range eviction / 8,266 |
| macos-x86_64 | 1（沿用） | 96.89 → 16.67 | 97.28% → 98.67% | 0.145 → 20.565 | 9,825.02 → 58.76 | Apple per-range eviction / 3,016 |
| macos-x86_64 | 2 | 97.33 → 16.97 | 97.28% → 98.67% | 0.106 → 40.524 | 24,614.41 → 59.04 | Apple per-range eviction / 3,031 |
| macos-x86_64 | 4 | 97.82 → 17.46 | 97.28% → 98.67% | 0.187 → 109.561 | 27,593.19 → 45.33 | Apple per-range eviction / 2,327 |
| macos-x86_64 | 8 | 98.68 → 18.46 | 97.28% → 98.67% | 0.202 → 222.139 | 23,383.89 → 44.29 | Apple per-range eviction / 2,274 |
| windows-arm64 | 1（沿用） | 95.11 → 18.62 | 97.28% → 98.67% | 0.087 → 5.057 | 14,818.25 → 238.67 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 12,251 |
| windows-arm64 | 2 | 95.89 → 19.51 | 97.28% → 98.67% | 0.125 → 11.876 | 20,524.54 → 207.24 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 10,638 |
| windows-arm64 | 4 | 96.90 → 20.00 | 97.28% → 98.67% | 0.109 → 22.233 | 44,486.43 → 211.93 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 10,879 |
| windows-arm64 | 8 | 97.65 → 21.08 | 97.28% → 98.67% | 0.117 → 50.715 | 42,378.75 → 190.14 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 9,760 |
| windows-x86_64 | 1（沿用） | 93.47 → 17.07 | 97.28% → 98.67% | 0.072 → 5.313 | 18,345.07 → 221.42 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 11,365 |
| windows-x86_64 | 2 | 94.01 → 17.49 | 97.28% → 98.67% | 0.115 → 14.721 | 25,376.16 → 159.65 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 8,195 |
| windows-x86_64 | 4 | 94.78 → 18.19 | 97.28% → 98.67% | 0.122 → 30.299 | 43,322.55 → 155.59 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 7,986 |
| windows-x86_64 | 8 | 95.77 → 19.35 | 97.28% → 98.67% | 0.124 → 64.769 | 42,970.89 → 148.46 | Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured / 7,621 |

移动端模拟器：

| 环境 | 查询线程 | RSS/工作集 MB | Recall@10 | P95 ms | QPS | DiskANN 缓存 / 实际 IOPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| android-x86_64-emulator **模拟器** | 1（沿用） | 95.21 → 15.64 | 97.28% → 98.67% | 0.086 → 8.190 | 16,880.48 → 143.91 | Android O_DIRECT requested / 7,387 |
| android-x86_64-emulator **模拟器** | 2 | 94.96 → 15.85 | 97.28% → 98.67% | 0.103 → 15.018 | 29,507.33 → 162.98 | Android O_DIRECT requested / 8,366 |
| android-x86_64-emulator **模拟器** | 4 | 95.82 → 16.34 | 97.28% → 98.67% | 0.102 → 30.781 | 29,413.29 → 157.48 | Android O_DIRECT requested / 8,083 |
| android-x86_64-emulator **模拟器** | 8 | 96.60 → 17.37 | 97.28% → 98.67% | 0.107 → 61.656 | 27,645.74 → 155.08 | Android O_DIRECT requested / 7,960 |
| ios-arm64-simulator **模拟器** | 1（沿用） | 132.10 → 50.68 | 97.28% → 98.67% | 0.263 → 11.903 | 8,396.37 → 154.62 | Apple per-range eviction / 7,937 |
| ios-arm64-simulator **模拟器** | 2 | 132.28 → 51.69 | 97.28% → 98.67% | 0.243 → 21.963 | 18,961.73 → 145.54 | Apple per-range eviction / 7,471 |
| ios-arm64-simulator **模拟器** | 4 | 132.35 → 52.22 | 97.28% → 98.67% | 0.281 → 35.837 | 21,218.01 → 195.25 | Apple per-range eviction / 10,022 |
| ios-arm64-simulator **模拟器** | 8 | 132.74 → 52.84 | 97.28% → 98.67% | 0.332 → 73.848 | 18,287.91 → 176.66 | Apple per-range eviction / 9,068 |

## 验证范围

- 本轮新增的 2/4/8 线程共用各自平台的同一个 CI 实例；各操作系统的 CPU、内存及宿主存储仍不同。
- Linux/Android 核对 OS 存储读字节；Android 为模拟器未加密目录的控制实验。
- Apple 为按范围驱逐页缓存，iOS 使用宿主对模拟器进程的计数；Windows 使用 FILE_FLAG_NO_BUFFERING。
- Windows 物理设备读及所有平台的设备/虚拟化宿主缓存未完全独立排除。
- 本轮复核 378 次新增查询进程试验；1 线程历史记录另存于每个平台的 baseline 字段。

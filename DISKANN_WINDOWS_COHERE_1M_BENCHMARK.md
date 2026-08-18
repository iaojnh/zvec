# Zvec DiskANN Windows 性能测试

## 背景

测试 Zvec DiskANN 在 Windows 下的性能表现，重点记录 Recall、QPS 和 RSS。

## 环境

- 服务器：阿里云 `ecs.g9i.4xlarge`
- 操作系统：Windows Server 2022，10.0.20348，64 位
- CPU：Intel64 Family 6 Model 106 Stepping 6，GenuineIntel
- 逻辑 CPU：16
- Python：3.12.7，MSC v.1941，AMD64
- 数据集：Cohere 1M
  - 训练数据：`D:\zvec_data\cohere_train_vector_1m.new.centaur.vecs`
  - 查询数据：`D:\zvec_data\cohere_test_vector_1000.new.txt`
  - Ground Truth：`D:\diskann_bench\baseline\ground_truth_d768_k100.txt`
- 代码分支：`feat/windows-diskann-benchmark`
- 代码提交：`e3af0baa`
- 构建类型：Release
- I/O 后端：`windows_overlapped`

> 本次测试记录的 Git 工作区状态为 Dirty。

## 结果

### 测试参数

| 项目 | 参数 |
| --- | --- |
| 测试数据 | 记录数：1,000,000；维度：768；查询数：1,000；距离度量：Cosine |
| 构建 | 线程：8；训练样本：200,000；max degree：32；builder list size：50；PQ chunks：384；memory limit：100.0 |
| 检索 | Cache 节点数：10,000；beam size：2；list size：100/300/500；线程：1/2/4；QPS TopK：50 |
| FP32 | `CosineFp32Converter` |
| FP16 | `CosineFp16Converter` |

### 测试数据

| 项目 | 指标 | FP32 L100 | FP32 L300 | FP32 L500 | FP16 L100 | FP16 L300 | FP16 L500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 构建 | 索引大小（GiB） | 4.18 |  |  | 2.28 |  |  |
| 构建 | 训练时间（s） | 232.182 |  |  | 464.040 |  |  |
| 构建 | 构建时间（s） | 311.084 |  |  | 320.433 |  |  |
| 构建 | 导出时间（s） | 8.095 |  |  | 6.102 |  |  |
| 构建 | 总 Wall 时间（s） | 552.976 |  |  | 791.823 |  |  |
| 构建 | 峰值 RSS（MiB） | 19,471.5 |  |  | 12,716.9 |  |  |
| Recall | Recall@1 | 92.900% | 96.500% | 97.800% | 92.400% | 96.900% | 98.100% |
| Recall | Recall@10 | 94.030% | 97.220% | 98.150% | 93.890% | 97.160% | 97.980% |
| Recall | Recall@50 | 91.424% | 96.644% | 97.880% | 91.370% | 96.498% | 97.728% |
| Recall | 峰值 RSS（MiB） | 455.9 | 457.0 | 456.0 | 438.7 | 439.8 | 440.5 |
| QPS | 1 线程 | 213.6 | 93.1 | 57.8 | 191.2 | 92.9 | 57.4 |
| QPS | 2 线程 | 245.7 | 93.1 | 57.6 | 245.1 | 92.9 | 57.5 |
| QPS | 4 线程 | 245.8 | 92.9 | 57.8 | 245.6 | 92.9 | 57.5 |
| 检索 RSS | 1 线程（MiB） | 422.6 | 422.8 | 422.7 | 407.2 | 407.1 | 407.1 |
| 检索 RSS | 2 线程（MiB） | 424.4 | 424.5 | 424.3 | 408.7 | 408.7 | 408.8 |
| 检索 RSS | 4 线程（MiB） | 427.7 | 428.0 | 428.0 | 411.8 | 411.9 | 412.3 |

> RSS 使用 Windows `PeakWorkingSetSize`，表示完整测试进程生命周期内的物理工作集峰值，包含索引加载和 Cache 预加载。

## 配置

### FP32 build.yaml

```yaml
BuilderCommon:
    BuilderClass: DiskAnnBuilder
    BuildFile: D:\zvec_data\cohere_train_vector_1m.new.centaur.vecs
    NeedTrain: true
    TrainFile: D:\zvec_data\cohere_train_vector_1m.new.centaur.vecs
    DumpPath: D:\diskann_bench\windows_report_20260818\indexes\diskann_fp32.index
    IndexPath: D:\diskann_bench\windows_report_20260818\indexes\diskann_fp32.index
    MetricName: Cosine
    ConverterName: CosineFp32Converter
    DisableIdMap: true
    ThreadCount: 8
    LogLevel: Info
BuilderParams:
    zvec.general.builder.thread_count: !!int 8
    zvec.diskann.builder.thread_count: !!int 8
    zvec.diskann.builder.max_degree: !!int 32
    zvec.diskann.builder.list_size: !!int 50
    zvec.diskann.builder.memory_limit: !!float 100.0
    zvec.diskann.builder.max_pq_chunk_num: !!int 384
    zvec.diskann.builder.max_train_sample_count: !!int 200000
```

### FP16 build.yaml

```yaml
BuilderCommon:
    BuilderClass: DiskAnnBuilder
    BuildFile: D:\zvec_data\cohere_train_vector_1m.new.centaur.vecs
    NeedTrain: true
    TrainFile: D:\zvec_data\cohere_train_vector_1m.new.centaur.vecs
    DumpPath: D:\diskann_bench\windows_report_20260818\indexes\diskann_fp16.index
    IndexPath: D:\diskann_bench\windows_report_20260818\indexes\diskann_fp16.index
    MetricName: Cosine
    ConverterName: CosineFp16Converter
    DisableIdMap: true
    ThreadCount: 8
    LogLevel: Info
BuilderParams:
    zvec.general.builder.thread_count: !!int 8
    zvec.diskann.builder.thread_count: !!int 8
    zvec.diskann.builder.max_degree: !!int 32
    zvec.diskann.builder.list_size: !!int 50
    zvec.diskann.builder.memory_limit: !!float 100.0
    zvec.diskann.builder.max_pq_chunk_num: !!int 384
    zvec.diskann.builder.max_train_sample_count: !!int 200000
```

### search.yaml 模板

Recall 测试的 `TopK` 为 `1,10,50`，QPS 测试的 `TopK` 为 `50`。脚本会分别替换 precision、list size 和线程数。

```yaml
SearcherCommon:
    SearcherClass: DiskAnnSearcher
    IndexPath: <FP32 或 FP16 索引路径>
    TopK: <1,10,50 或 50>
    QueryFile: D:\zvec_data\cohere_test_vector_1000.new.txt
    QueryType: float
    QueryFirstSep: ";"
    QuerySecondSep: " "
    GroundTruthFile: D:\diskann_bench\baseline\ground_truth_d768_k100.txt
    RecallThreadCount: 16
    RecallGTCount: 100
    RecallScorePrecision: 1e-4
    BenchThreadCount: <1、2 或 4>
    BenchSecs: 30
    BenchIterCount: 10000000
    CompareById: true
    ContainerType: FileReadStorage
    LogLevel: Info
SearcherParams:
    zvec.diskann.searcher.cache_node_num: !!int 10000
    zvec.diskann.searcher.list_size: !!int <100、300 或 500>
    zvec.diskann.searcher.beam_size: !!int 2
ContainerParams: {}
```

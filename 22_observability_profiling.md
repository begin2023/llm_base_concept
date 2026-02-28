# 22. 可观测性与性能分析

---

## 一、为什么推理系统需要可观测性

LLM 推理系统是复杂的分布式系统，涉及多个组件：GPU 执行、KV Cache 管理、调度、网络通信等。没有完善的可观测性，难以：

- 定位性能瓶颈（是 Attention 慢？还是 FFN 慢？还是调度问题？）
- 发现资源浪费（GPU 利用率低？内存泄漏？）
- 保证 SLA（首 Token 延迟超标？吞吐量不足？）
- 排查线上故障

---

## 二、关键指标（Metrics）

### 2.1 延迟指标

**TTFT（Time To First Token，首 Token 延迟）**：
- 从请求到达到第一个 token 生成的时间
- 主要由 Prefill 时间决定
- SLA 通常要求 P99 TTFT < 1-5 秒

**TPOT（Time Per Output Token，每 Token 延迟）**：
- 生成每个后续 token 的平均时间
- 决定了流式输出的"感知速度"
- SLA 通常要求 P99 TPOT < 50-100 ms

**E2E Latency（端到端延迟）**：
- 从请求到完成所有 token 的总时间
- = TTFT + TPOT × (num_output_tokens - 1)

**ITL（Inter-Token Latency，逐 token 延迟）**：
- 相邻两个 output token 之间的时间间隔
- 反映 decode 阶段的稳定性，抖动（jitter）也很重要

### 2.2 吞吐量指标

**Throughput（吞吐量）**：
- Output tokens/second（生成速度）
- Requests/second（请求处理速度）

**GPU Utilization（GPU 利用率）**：
- SM Utilization（流多处理器利用率）
- Memory Bandwidth Utilization（内存带宽利用率）
- 推理中 decode 阶段通常 SM 利用率低但带宽利用率高

### 2.3 内存指标

**KV Cache 使用率**：
- 已分配 KV Cache blocks / 总 KV Cache blocks
- 过高会导致请求被抢占（preemption），影响性能

**GPU 显存占用**：
- 模型权重占用
- KV Cache 占用
- Activation 占用

**CPU 内存占用**（如果有 swap）：
- 被换出的 KV Cache 大小

### 2.4 调度指标

**Waiting Queue Length（等待队列长度）**：
- 等待被处理的请求数量
- 过大说明系统过载

**Batch Size（批大小）**：
- 每次 iteration 处理的请求数
- 反映系统利用率

**Preemption Rate（抢占率）**：
- 因内存不足被抢占的请求比例
- 过高说明内存配置不合理

---

## 三、Prometheus + Grafana 监控体系

### 3.1 vLLM 的 Metrics

vLLM 内置了 Prometheus 格式的 metrics 端点（默认 `/metrics`）：

```python
# 启动时开启 metrics
python -m vllm.entrypoints.openai.api_server \
    --model ... \
    --enable-metrics \
    --metrics-endpoint /metrics
```

关键 metrics（Prometheus 格式）：

```
# 吞吐量
vllm:num_requests_running        # 当前运行中的请求数
vllm:num_requests_waiting        # 等待队列中的请求数
vllm:num_requests_swapped        # 被换出的请求数

# 吞吐量计数
vllm:prompt_tokens_total         # 处理的 prompt token 总数
vllm:generation_tokens_total     # 生成的 token 总数

# 延迟（histogram）
vllm:time_to_first_token_seconds # TTFT 分布
vllm:time_per_output_token_seconds # TPOT 分布
vllm:e2e_request_latency_seconds # 端到端延迟分布

# 内存
vllm:gpu_cache_usage_perc        # GPU KV Cache 使用率
vllm:cpu_cache_usage_perc        # CPU KV Cache 使用率（swap）

# GPU
vllm:gpu_cache_hit_rate          # KV Cache 命中率（Prefix Caching 时）
```

### 3.2 SGLang 的 Metrics

SGLang 同样提供 Prometheus metrics：

```
sglang:num_running_reqs          # 运行中的请求数
sglang:num_waiting_reqs          # 等待的请求数
sglang:token_usage               # Token 使用率
sglang:cache_hit_rate            # RadixCache 命中率
sglang:avg_prompt_len            # 平均 prompt 长度
sglang:avg_gen_len               # 平均生成长度
```

### 3.3 Grafana Dashboard

搭建监控 Dashboard 的关键面板：

```yaml
推荐 Dashboard 面板布局：

Row 1: 实时概览
  - 当前 QPS
  - P50/P90/P99 TTFT
  - P50/P90/P99 TPOT
  - GPU 利用率

Row 2: 吞吐量
  - Tokens/s（时间序列）
  - Requests/s（时间序列）
  - 活跃请求数 / 等待队列长度

Row 3: 内存
  - KV Cache 使用率（时间序列）
  - GPU 显存占用
  - 抢占率（如果有）

Row 4: 详细延迟分布
  - TTFT 分位数（P50/P90/P99/P999）
  - TPOT 分位数
  - E2E 延迟分位数
```

---

## 四、GPU 性能分析工具

### 4.1 NVIDIA Nsight Systems（nsys）

Nsight Systems 是分析 GPU 时间线的主要工具：

```bash
# 基本使用
nsys profile \
    --output profile_output \
    --trace cuda,nvtx,osrt \
    python my_inference_script.py

# 查看结果（GUI）
nsys-ui profile_output.nsys-rep

# 命令行报告
nsys stats profile_output.nsys-rep
```

关键功能：
- **Timeline View**：可视化 CPU/GPU 操作的时间线
- **CUDA API Trace**：每个 CUDA API 调用的耗时
- **Kernel Duration**：每个 GPU kernel 的执行时间
- **Memory Transfer**：CPU-GPU 数据传输时间
- **NVTX 标注**：在代码中添加标记，在 Timeline 中显示

在 vLLM/SGLang 中添加 NVTX 标注：
```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("prefill_attention")
output = attention_layer(hidden_states)
nvtx.range_pop()
```

### 4.2 NVIDIA Nsight Compute（ncu）

Nsight Compute 分析单个 kernel 的性能：

```bash
# 分析指定 kernel
ncu --kernel-name "flash_attention_kernel" \
    --metrics "sm__throughput,dram__throughput" \
    python my_script.py
```

关键指标：
- **Roofline Analysis**：判断 kernel 是 compute-bound 还是memory-bound
- **Memory Access Pattern**：L1/L2 cache 命中率
- **Warp Efficiency**：Warp 的有效执行率
- **Occupancy**：SM 上活跃 Warp 的数量

### 4.3 PyTorch Profiler

PyTorch 内置的 Profiler，更易用：

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        output = model(input_ids)

# 打印 CPU + CUDA 耗时 top
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# 导出 Chrome Trace（在浏览器中查看）
prof.export_chrome_trace("trace.json")
# 用 chrome://tracing 或 Perfetto UI 查看

# 导出 TensorBoard
prof.export_stacks("profiler_stacks.txt", metric="self_cuda_time_total")
```

### 4.4 torch.cuda 内置工具

```python
# 计时 GPU 操作
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(input_ids)
end.record()

torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # 毫秒
print(f"Inference time: {elapsed:.2f} ms")

# 内存统计
print(torch.cuda.memory_summary())
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 内存快照（用于定位内存泄漏）
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
```

---

## 五、分布式推理的可观测性

### 5.1 多 GPU 场景

在 Tensor Parallelism 场景下，需要跟踪：

```python
# 在每个 rank 上收集 metrics
import torch.distributed as dist

local_rank = dist.get_rank()
# 在 rank 0 上汇总
if local_rank == 0:
    # 发布 metrics
    pass
```

**AllReduce 通信分析**：
- Nsight Systems 可以显示 NCCL 通信操作的时间
- 关键 NCCL 操作：AllReduce（TP 同步）、All-to-All（MoE Expert 路由）

### 5.2 OpenTelemetry 分布式追踪

对于多节点 PD 分离系统：

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 配置 Tracer
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm_inference")

# 在代码中添加 Span
with tracer.start_as_current_span("prefill") as span:
    span.set_attribute("prompt_len", len(input_ids))
    span.set_attribute("batch_size", batch_size)
    output = prefill_model(input_ids)
```

---

## 六、推理框架专用分析工具

### 6.1 vLLM 内置分析

```python
# 使用 vLLM 的 benchmark 工具
python -m vllm.benchmark.benchmark_throughput \
    --model meta-llama/Llama-2-7b \
    --num-prompts 1000 \
    --request-rate 10 \
    --output-json results.json

# 分析调度行为
python -m vllm.benchmark.benchmark_latency \
    --model meta-llama/Llama-2-7b \
    --batch-size 1 \
    --input-len 1024 \
    --output-len 128
```

### 6.2 SGLang 的 benchmark

```bash
# SGLang benchmark
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 1000 \
    --request-rate 20 \
    --model meta-llama/Llama-3-8B
```

### 6.3 常用 benchmark 数据集

- **ShareGPT**：真实的用户对话数据，prompt/response 长度分布真实
- **Alpaca**：指令微调数据集，短 prompt 为主
- **MT-Bench**：多轮对话评测
- **GSM8K / MATH**：推理能力评测

---

## 七、常见性能问题诊断

### 7.1 GPU 利用率低

**症状**：`gpu_util` < 50%，但系统负载正常

**可能原因**：
1. Batch size 太小（decode 阶段常见）→ 增加并发请求
2. CUDA Graph 未启用（大量 kernel launch overhead）→ 启用 CUDA Graph
3. CPU 成为瓶颈（采样、调度慢）→ 优化 CPU 侧代码

### 7.2 KV Cache 频繁 OOM / 抢占

**症状**：`preemption_rate` 高，等待队列长

**可能原因**：
1. KV Cache 分配过小 → 调整 `gpu-memory-utilization`
2. 请求太长（prompt + generation 超过预期）→ 限制最大序列长度
3. Prefix Caching 未命中，缓存利用率低 → 检查工作负载特征

### 7.3 TTFT 过高

**症状**：P99 TTFT >> P50 TTFT

**可能原因**：
1. Prefill 阶段被长 decode batch 阻塞 → 考虑 PD 分离
2. Chunked Prefill 分块过小 → 调整 chunk size
3. 等待队列积压 → 增加并发处理能力

---

## 八、总结

LLM 推理系统的可观测性体系需要覆盖三个层次：

1. **业务层**：TTFT、TPOT、E2E Latency、Throughput
   - 工具：Prometheus + Grafana

2. **系统层**：GPU 利用率、KV Cache 使用率、调度指标
   - 工具：vLLM/SGLang 内置 metrics，Nsight Systems

3. **算子层**：单个 kernel 性能、内存访问模式、通信开销
   - 工具：Nsight Compute、PyTorch Profiler、NCCL 分析

完善的可观测性是推理系统稳定运行和持续优化的基础。

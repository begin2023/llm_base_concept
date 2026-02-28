# 38. 显存估算与模型部署规划

---

## 一、显存组成分析

在 LLM 推理中，GPU 显存主要由以下几部分组成：

```
总显存 = 模型权重显存
       + KV Cache 显存
       + Activation 显存（推理时相对较小）
       + CUDA 上下文 / 框架开销
       + 其他（临时缓冲区等）
```

---

## 二、模型权重显存计算

### 2.1 基本公式

$$\text{模型权重显存} = \text{参数量} \times \text{每参数字节数}$$

| 数据类型 | 每参数字节数 | 说明 |
|---------|-----------|------|
| FP32 | 4 bytes | 全精度 |
| FP16 / BF16 | 2 bytes | 半精度，推理常用 |
| INT8 | 1 byte | 8-bit 量化 |
| INT4 / NF4 | 0.5 bytes | 4-bit 量化 |
| FP8 | 1 byte | 8-bit 浮点，新型量化 |

### 2.2 常见模型显存估算

| 模型 | 参数量 | BF16 | INT8 | INT4 |
|------|--------|------|------|------|
| Qwen2.5-7B | 7.6B | ~15 GB | ~7.6 GB | ~3.8 GB |
| Llama-3-8B | 8.0B | ~16 GB | ~8 GB | ~4 GB |
| Llama-3-70B | 70B | ~140 GB | ~70 GB | ~35 GB |
| Qwen2.5-72B | 72B | ~144 GB | ~72 GB | ~36 GB |
| DeepSeek V3 | 671B | ~1342 GB | ~671 GB | ~336 GB |
| DeepSeek R1 | 671B | ~1342 GB | ~671 GB | ~336 GB |

**快速估算公式**：
- BF16：参数量（B） × 2 GB
- INT8：参数量（B） × 1 GB
- INT4：参数量（B） × 0.5 GB

---

## 三、KV Cache 显存计算

### 3.1 KV Cache 大小公式

$$\text{KV Cache 总大小} = 2 \times N_\text{layers} \times N_\text{kv\_heads} \times d_\text{head} \times L_\text{seq} \times \text{dtype\_bytes}$$

其中：
- $N_\text{layers}$：Transformer 层数
- $N_\text{kv\_heads}$：KV Head 数（GQA 时小于 Q Head 数）
- $d_\text{head}$：每个 head 的维度（= hidden_dim / num_heads）
- $L_\text{seq}$：序列长度（= prompt_len + generation_len）
- 系数 2：K 和 V 各一份

### 3.2 各模型 KV Cache 大小（每 token，BF16）

| 模型 | layers | kv_heads | head_dim | 每 token KV Cache |
|------|--------|----------|----------|------------------|
| Llama-3-8B | 32 | 8 | 128 | 2×32×8×128×2 = 131 KB |
| Llama-3-70B | 80 | 8 | 128 | 2×80×8×128×2 = 327 KB |
| Qwen2.5-7B | 28 | 4 | 128 | 2×28×4×128×2 = 57 KB |
| Llama-2-7B | 32 | 32 | 128 | 2×32×32×128×2 = 524 KB |

### 3.3 KV Cache 总量计算

```
KV Cache 总量 = 每 token KV Cache × 并发请求数 × 平均序列长度

示例：Llama-3-8B
  每 token KV Cache = 131 KB
  并发请求数 = 100
  平均序列长度 = 2048 tokens

  KV Cache = 131 KB × 100 × 2048 = 26.8 GB
```

---

## 四、完整显存规划

### 4.1 推理显存组成

```
总显存 = 模型权重
       + KV Cache
       + 激活值（Activation）
       + CUDA/框架开销

Activation 显存（推理）：
  ≈ batch_size × max_seq_len × hidden_dim × num_layers × dtype_bytes
  （比训练小得多，因为不需要存梯度）
  通常约为模型权重的 5-10%

CUDA/框架开销：1-3 GB（CUDA context、库加载等）
```

### 4.2 vLLM 的显存分配策略

vLLM 使用 `gpu_memory_utilization` 参数控制显存分配：

```
vLLM 显存分配：
  总 GPU 显存 = 模型权重 + KV Cache Pool

  KV Cache Pool = 总显存 × gpu_memory_utilization - 模型权重 - 预留

  例如：A100 80GB，gpu_memory_utilization=0.9，模型权重 16GB
  KV Cache Pool = 80 × 0.9 - 16 - 2（系统预留）= 54 GB

  支持的最大并发 tokens = 54 GB / (每 token KV Cache 大小)
```

### 4.3 实际部署规划工具

**vLLM 的预估工具**：

```python
from vllm import LLM

# 不实际加载模型，只估算显存
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.9,
    dtype="bfloat16",
    # dry_run=True  # 仅估算，不实际运行（某些版本支持）
)

# 查看 KV Cache 统计
stats = llm.get_engine_stats()
```

---

## 五、不同硬件的部署规划

### 5.1 单卡部署

| GPU | 显存 | 能运行的模型（BF16） |
|-----|------|-------------------|
| RTX 3090/4090 | 24 GB | ≤ 13B |
| A10G | 24 GB | ≤ 13B |
| A40 | 48 GB | ≤ 30B |
| A100 40GB | 40 GB | ≤ 20B |
| A100 80GB | 80 GB | ≤ 70B（紧张） |
| H100 80GB | 80 GB | ≤ 70B |
| H100 NVL | 94 GB | ≤ 70B（充裕） |

### 5.2 多卡 Tensor Parallel 部署

部署 70B 模型的方案：

| 方案 | GPU 数量 | 单卡显存需求 | 说明 |
|------|---------|-----------|------|
| TP=4，A100 80GB | 4 | 35 GB（够用）| 推荐方案 |
| TP=2，A100 80GB | 2 | 70 GB（紧张）| KV Cache 空间小 |
| TP=8，A100 40GB | 8 | 17.5 GB | 通信开销较大 |
| TP=4，H100 80GB | 4 | 35 GB | 性能更好 |

部署 671B DeepSeek V3（BF16）：
```
总权重显存 = 671B × 2 bytes = 1342 GB

方案1：8× H100 80GB（640GB 总显存）+ INT4 量化
  INT4 权重 = 671B × 0.5 bytes = 336 GB
  8× H100 = 640 GB 总显存
  KV Cache = 640 - 336 - 开销 = ~280 GB

方案2：16× H100 80GB（1280GB 总显存）+ BF16
  16× H100 = 1280 GB
  模型权重 = 1342 GB → 不够，需要更多卡或量化

方案3：32× H100 80GB + BF16
  实际 DeepSeek 内部部署使用了专用集群
```

### 5.3 vLLM 多卡部署命令

```bash
# 4 卡 TP 部署 Llama-3-70B（BF16）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --port 8000

# 使用量化部署（节省显存）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 2 \
    --quantization awq \  # 或 gptq, fp8
    --gpu-memory-utilization 0.9
```

---

## 六、吞吐量与显存的权衡

### 6.1 KV Cache 大小对吞吐量的影响

更大的 KV Cache Pool → 更多并发请求 → 更高吞吐量：

```
A100 80GB，Llama-3-8B，BF16：
  模型权重 = 16 GB
  可用 KV Cache = 80 × 0.9 - 16 = 56 GB
  每 token KV Cache = 131 KB
  最大并发 tokens = 56 GB / 131 KB = ~437,000 tokens

  如果平均序列长度 = 2048 tokens：
  最大并发请求 ≈ 437000 / 2048 ≈ 213 个请求

  实测峰值吞吐量：约 3000-4000 output tokens/s
```

### 6.2 量化对吞吐量的影响

| 量化方式 | 显存节省 | 吞吐量影响 | 质量损失 |
|---------|---------|----------|---------|
| BF16（基准） | 0% | 基准 | 无 |
| FP8 | ~50% | +10-20% | 极小（<1%） |
| INT8 AWQ | ~50% | ~基准 | 小（1-2%） |
| INT4 GPTQ | ~75% | -10-20%（权重加载慢）| 中等（2-5%） |

**FP8 是当前最推荐的量化方案**（H100 原生支持 FP8 矩阵乘法，显存节省 + 速度提升 + 质量几乎无损）。

---

## 七、部署规划实操

### 7.1 业务需求 → 硬件配置

**给定业务需求**：
- 模型：Llama-3-70B
- QPS：50 requests/s
- 平均 prompt 长度：1000 tokens
- 平均生成长度：500 tokens
- P99 TTFT < 2s，P99 TPOT < 100ms

**计算过程**：

```
Step 1: 确定模型显存需求
  Llama-3-70B BF16 = 140 GB

Step 2: 确定 KV Cache 需求
  每 token KV Cache（Llama-3-70B）= 327 KB
  平均序列长度 = 1000 + 500 = 1500 tokens
  每请求 KV Cache = 327 KB × 1500 = 490 MB

  并发请求数估算：
  QPS=50，平均响应时间=3s（TTFT+生成时间）
  并发请求 ≈ 50 × 3 = 150 个

  KV Cache 总量 = 490 MB × 150 = 73.5 GB

Step 3: 总显存需求
  总显存 = 140 GB（权重）+ 73.5 GB（KV Cache）+ 5 GB（其他）
         = ~220 GB

Step 4: 硬件选型
  方案1：4× H100 80GB（TP=4）
    总显存 = 320 GB > 220 GB ✓
    TP=4 通信开销可接受

  方案2：4× A100 80GB（TP=4）
    总显存 = 320 GB > 220 GB ✓
    性能比 H100 略低

  方案3：使用 FP8 量化，节省一半显存
    Llama-3-70B FP8 = 70 GB
    总显存 = 70 + 73.5 + 5 = ~150 GB
    → 2× H100 80GB（TP=2）即可
```

### 7.2 常用显存估算工具

```python
# 方法1: 直接计算
def estimate_vram(
    num_params_b,           # 参数量（十亿）
    dtype_bytes=2,          # BF16
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    max_batch_tokens=50000, # 最大并发 token 数
):
    model_vram = num_params_b * 1e9 * dtype_bytes / (1024**3)  # GB
    kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes / 1024  # KB
    kv_cache_total = kv_cache_per_token * max_batch_tokens / (1024**2)  # GB
    return model_vram + kv_cache_total + 3  # +3 GB 系统开销

print(f"Llama-3-8B 显存需求: {estimate_vram(8):.1f} GB")
print(f"Llama-3-70B 显存需求: {estimate_vram(70, num_layers=80):.1f} GB")
```

---

## 八、总结

显存规划的核心公式：

```
推理显存 = 模型权重显存 + KV Cache 显存 + 系统开销

模型权重 = 参数量 × 每参数字节数
  BF16: ×2, INT8: ×1, INT4: ×0.5, FP8: ×1

KV Cache（per token）= 2 × layers × kv_heads × head_dim × dtype_bytes

部署决策树：
1. 能否单卡放下权重？
   - 能 → 考虑量化进一步节省显存
   - 不能 → 使用 TP（Tensor Parallel）
2. KV Cache 够用吗（满足并发需求）？
   - 够 → 当前配置可行
   - 不够 → 增加 GPU 数量，或减少最大序列长度
3. 吞吐量是否满足 QPS 需求？
   - 满足 → 当前配置可行
   - 不满足 → 增加 GPU 实例（Data Parallel）
```

掌握显存估算能力是 LLM 系统工程师的核心技能，直接影响部署成本和系统性能。

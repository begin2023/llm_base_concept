# 31. 长上下文推理（Ring Attention、序列并行）

---

## 一、长上下文推理的挑战

随着 LLM 上下文窗口不断扩展（Claude 200K、Gemini 1M、Kimi 128K+），长上下文推理带来了独特的工程挑战：

### 1.1 显存挑战

**KV Cache 随序列长度线性增长**：

```
KV Cache 大小 = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes

Llama-3-8B（GQA, BF16），seq_len = 128K：
= 2 × 32 × 8 × 128 × 131072 × 2
= 17.2 GB（仅 KV Cache）

+ 模型权重 ~16 GB
= 需要 33+ GB 显存（超过单卡 A100）
```

### 1.2 计算挑战

**Attention 计算复杂度为 O(seq_len²)**：

```
标准 Attention FLOPS = 2 × batch × num_heads × seq_len² × head_dim

seq_len = 128K：
FLOPS ∝ (128000)² = 1.6 × 10¹⁰

相比 seq_len = 4K：(4000)² = 1.6 × 10⁷
→ 长度增加 32×，Attention 计算增加 32² = 1024×
```

这使得长序列推理极其耗时且需要大量 GPU 内存。

---

## 二、序列并行（Sequence Parallelism，SP）

序列并行是将长序列的 Attention 计算分布到多个 GPU 的技术。

### 2.1 基本思想

将序列按 token 维度切分：

```
序列长度 L = 131072 tokens，4 GPU SP：
GPU 0: token[0:32768]
GPU 1: token[32768:65536]
GPU 2: token[65536:98304]
GPU 3: token[98304:131072]
```

每个 GPU 只存储部分 KV Cache，显存需求降低为原来的 1/N。

### 2.2 序列并行的 Attention 问题

Attention 需要 Query 与所有 Key/Value 做点积，但在序列并行中，Key 和 Value 分布在不同 GPU 上。

**解决方案：AllGather K/V**

```
每个 GPU 有自己的 Q_i（本地 Query）和 K_i, V_i（本地 Key/Value）

Step 1: AllGather K 和 V → 每个 GPU 都获得完整的 K, V
Step 2: 每个 GPU 用自己的 Q_i 与完整的 K, V 做 Attention
Step 3: ReduceScatter 输出 → 每个 GPU 得到自己负责的 token 的输出

通信量：2 × L × num_heads × head_dim（AllGather K 和 V）
```

---

## 三、Ring Attention

Ring Attention（Liu et al. 2023）是更高效的序列并行 Attention 实现，通过流水线化隐藏通信开销。

### 3.1 核心思想

将 GPU 排成一个"环"，每个 GPU 将自己的 K/V 块沿环传递，同时在本地计算 Attention。

```
GPU 排列：GPU0 → GPU1 → GPU2 → GPU3 → GPU0（环形）

Step 1: 每个 GPU 计算本地 K/V 与本地 Q 的 Attention
Step 2: GPU i 将 K/V 块传给 GPU (i+1)%N，同时接收 GPU (i-1)%N 的 K/V
Step 3: 计算接收到的 K/V 与本地 Q 的 Attention，累积
...
重复 N 步，每个 GPU 获得完整的 Attention 输出
```

### 3.2 Ring Attention 的流水线优化

关键优化：**计算与通信重叠（Overlap）**

```
时间轴：
GPU 0:  [计算 K/V_0] [发送 K/V_0 到 GPU1]
             ↕ 同时进行
             [接收 K/V_3] [计算 K/V_3]
             ↕ 同时进行
             [发送 K/V_3 到 GPU1] [接收 K/V_2] [计算 K/V_2]
             ...
```

通过 PyTorch 的异步通信（`torch.distributed.isend/irecv`）或 NCCL 的非阻塞操作，传输时间可以被计算时间完全覆盖，实现近似线性扩展。

### 3.3 Ring Attention 的 FlashAttention 集成

Ring Attention 与 FlashAttention 结合（称为 FlashAttention Ring）：

- 每个 GPU 上的局部 Attention 使用 FlashAttention 计算（内存高效）
- 利用 FlashAttention 的 Online Softmax 特性，正确累积跨 GPU 的 Attention 分数

**Online Softmax 的关键**：Softmax 的分母（归一化常数）需要跨所有 K/V 块累积，Ring Attention 通过维护运行中的最大值和归一化常数来实现正确的累积。

---

## 四、长上下文推理的其他技术

### 4.1 DejaVu 注意力（稀疏注意力）

对于超长序列，并非所有 token 都需要完整的 Attention。稀疏 Attention 方法：

**Local Window Attention（局部窗口注意力）**：
- 每个 token 只 attend 到最近的 W 个 token
- 复杂度降至 O(L × W)
- 损失了全局信息

**Sliding Window + Global Tokens（Mistral/Longformer）**：
```
local: 每个 token attend 到 ±W 范围内的 token
global: 少数 global token（如 [CLS]）attend 到所有 token
```

### 4.2 StreamingLLM（无限长流式推理）

StreamingLLM 解决了无限长序列推理问题：

**问题**：当序列超过 context window 时，传统方法会失效。

**方案**：保留"Attention Sink" token（最初的几个 token，通常是 1-4 个）和最近的 W 个 token 的 KV Cache，丢弃中间的 KV Cache：

```
全部 KV Cache: [tok_0, tok_1, ..., tok_k, tok_{k+1}, ..., tok_n]
StreamingLLM:   [tok_0, tok_1(sink)] + [tok_{n-W}, ..., tok_n]
                 ↑ Attention Sink      ↑ 滑动窗口
```

**为什么保留 Attention Sink？**
- 语言模型会将"全局"信息偏向存储在开头 token 的 KV Cache 中
- 丢弃开头 token 会导致模型性能崩溃
- 只需保留 1-4 个"sink"就足够

### 4.3 LongLoRA

LongLoRA 用极低成本扩展模型的上下文窗口：

**Shifted Sparse Attention（S²-Attn）**：
- 训练时将 token 分组，组内使用全注意力
- 通过 shift（错位分组）让相邻组的信息传递
- 显著降低训练时的显存需求

---

## 五、KV Cache 压缩技术

对于长上下文，KV Cache 占用大量显存，需要压缩：

### 5.1 KV Cache 量化

```python
# 将 KV Cache 量化为 INT8
# 推理时用 INT8 存储，计算 Attention 时反量化
kv_cache_fp16 = compute_kv(hidden_states)
kv_cache_int8 = quantize(kv_cache_fp16, scale, zero_point)
# 存储 int8，节省 50% 显存
```

### 5.2 KV Cache 剪枝（Eviction）

**H2O（Heavy Hitters Oracle）**：
- 统计哪些 KV Cache token 被频繁访问（"Heavy Hitters"）
- 保留这些重要 token 的 KV Cache，丢弃不重要的
- 可将 KV Cache 大小压缩到原来的 20%，性能损失很小

**SnapKV**：
- 在 Prefill 阶段，分析 Attention 权重，选择最重要的 KV Cache token 保留
- 对于超长 prompt 效果显著

### 5.3 MLA（Multi-head Latent Attention）

DeepSeek V2/V3 的 MLA 通过将 KV 压缩为低维潜在向量，大幅减少 KV Cache 大小（详见第 26 章）。

---

## 六、在推理框架中的支持

### 6.1 vLLM 的长上下文支持

```python
from vllm import LLM

# 启用长上下文（需要足够显存）
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=131072,            # 128K context
    tensor_parallel_size=4,          # TP=4 分担 KV Cache
    gpu_memory_utilization=0.95,     # 最大化 KV Cache
    enable_chunked_prefill=True,     # Chunked Prefill（避免 OOM）
)
```

### 6.2 SGLang 的长上下文支持

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --context-length 131072 \
    --tp 4 \
    --chunked-prefill-size 4096
```

---

## 七、总结

长上下文推理的核心技术：

| 技术 | 解决的问题 | 核心机制 |
|------|---------|---------|
| 序列并行（SP） | 单卡显存不足 | 将序列切分到多 GPU |
| Ring Attention | SP 通信开销 | 流水线化通信，计算覆盖传输 |
| StreamingLLM | 超出 context window | Attention Sink + 滑动窗口 |
| KV Cache 量化 | KV Cache 显存占用 | INT8/INT4 量化 |
| KV Cache 剪枝 | KV Cache 显存占用 | 保留重要 KV，丢弃不重要的 |
| Sparse Attention | O(L²) 计算复杂度 | 局部窗口 + 全局 token |
| MLA | KV Cache 大小 | 低维潜在向量压缩 |

长上下文推理是当前 LLM 系统研究的热点方向，随着模型上下文窗口不断扩大，高效的序列并行和 KV Cache 管理将愈发重要。

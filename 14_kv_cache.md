# 14. KV Cache 管理 详解

## 一、KV Cache 的概念

### 1.1 什么是 KV Cache

KV Cache（Key-Value Cache）是大模型推理中最核心的优化技术之一。它的核心思想是：**缓存已计算的 Key 和 Value 张量，避免在自回归生成过程中重复计算**。

#### 1.1.1 为什么需要 KV Cache

Transformer 的自回归生成过程：
```
步骤 1：输入 "The"        → 计算 Q1, K1, V1 → Attention(Q1, [K1], [V1]) → 输出 "cat"
步骤 2：输入 "The cat"    → 计算 Q2, K2, V2 → Attention(Q2, [K1,K2], [V1,V2]) → 输出 "sat"
步骤 3：输入 "The cat sat" → 计算 Q3, K3, V3 → Attention(Q3, [K1,K2,K3], [V1,V2,V3]) → 输出 "on"
...

问题：每一步都需要重新计算之前所有 token 的 K 和 V！
步骤 N：需要计算 N 个 token 的 K 和 V，但其中 N-1 个已经在之前的步骤中计算过了。
```

**Without KV Cache（无缓存）**：
- 每一步都是 O(N) 的 K/V 计算 + O(N²) 的 attention 计算
- 生成 T 个 token 的总计算量：O(T × N²) → 非常慢！

**With KV Cache（有缓存）**：
- 只需要计算第 N 个 token 的 Q_N, K_N, V_N（O(1) 的 projection）
- K_N, V_N 追加到缓存中
- 生成 T 个 token 的总计算量：O(T × N) → 大幅降低！

#### 1.1.2 KV Cache 的数据结构

```python
# 每一层 Transformer 都有自己的 KV Cache
# 以 LLaMA-2-7B 为例：32 层，32 个 attention head（GQA 8 个 KV head），head_dim=128

# 单个 token 在一层中的 KV Cache:
K_cache_per_layer_per_token = [num_kv_heads, head_dim]  # [8, 128] for GQA
V_cache_per_layer_per_token = [num_kv_heads, head_dim]  # [8, 128] for GQA

# 单个请求的完整 KV Cache 大小公式：
# 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

---

## 二、显存占用分析

### 2.1 KV Cache 显存计算公式

$$\text{KV Cache} = 2 \times L \times H_{kv} \times D \times S \times \text{sizeof(dtype)}$$

其中：
- $2$ = Key + Value
- $L$ = Transformer 层数
- $H_{kv}$ = KV head 数（MHA 中 = num_heads，GQA 中更少，MQA 中 = 1）
- $D$ = head dimension
- $S$ = 序列长度

### 2.2 典型模型的 KV Cache 显存计算

**LLaMA-2-7B（MHA，32 heads）**：
```
= 2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes ≈ 0.5 MB per token
1024 tokens 的请求 = 0.5 MB × 1024 = 512 MB
Batch=16, 1024 tokens = 512 MB × 16 = 8 GB
```

**LLaMA-3-8B（GQA，8 KV heads）**：
```
= 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 0.125 MB per token
GQA 将 KV Cache 减少了 4 倍！（32/8 = 4）
```

**DeepSeek-V3（MLA，压缩 KV）**：
```
MLA 将 KV 压缩到 512 维潜在空间（而非 num_heads × head_dim）
= 61 × 512 × 2 bytes = 62,464 bytes ≈ 0.06 MB per token

对比：MHA ~0.5 MB/token，GQA ~0.125 MB/token，MLA ~0.06 MB/token
```

### 2.3 各因素对 KV Cache 显存的影响

| 因素 | 影响 | 优化方向 |
|------|------|---------|
| 序列长度 S | 线性增长 | Paged Attention、KV Cache 压缩 |
| Batch Size B | 线性增长 | Prefix Caching 共享公共前缀 |
| 层数 L | 线性增长 | 模型架构选择 |
| KV Head 数 H_kv | 线性增长 | GQA、MQA、MLA |
| 数据类型 | 线性影响 | FP8/INT8 量化（减半） |

---

## 三、管理策略

### 3.1 Paged Attention（参见第 11 章详细介绍）

核心思想回顾：
- 将 KV Cache 分成固定大小的 Block（如 16 tokens）
- 通过 Block Table 实现逻辑块到物理块的映射
- 按需分配，消除外部碎片，减少内部碎片
- 支持 Copy-on-Write（写时复制）

### 3.2 预分配策略 vs 动态分配策略

| 策略 | 说明 | 优缺点 |
|------|------|--------|
| 预分配（Pre-allocation） | 每个请求一次性分配 max_seq_len 的空间 | 简单但浪费严重 |
| 动态分配（Paged Attention） | 按需分配，生成一个 token 才分配空间 | 显存利用率高，是现代框架标准 |

---

## 四、KV Cache 量化

### 4.1 为什么要量化 KV Cache

KV Cache 占用的显存可能远超模型权重，量化 KV Cache 可以：
- 减少显存占用，支持更多并发请求
- 减少内存带宽消耗（Attention 计算是 memory-bound 的）
- 提升 decode 阶段的吞吐量

### 4.2 FP8 KV Cache

```
FP16 → FP8: 显存占用减半

FP8 有两种格式：
  E4M3（4 位指数，3 位尾数）：精度更高，范围较小，适合推理
  E5M2（5 位指数，2 位尾数）：范围更大，精度较低

FP8 KV Cache 实现：
  Prefill 阶段：计算 K, V 后，量化为 FP8 存入缓存
  Decode 阶段：从缓存读取 FP8 的 K, V，在 Attention kernel 内部反量化后计算

精度影响：FP8 E4M3 KV Cache 对大多数模型的影响极小（PPL 增加 < 0.1%）
```

### 4.3 INT8 KV Cache

- **Per-tensor 量化**：精度较差，整个张量共享一个 scale
- **Per-channel 量化**：每个 head 有独立的 scale，精度更好
- **Per-token 量化**：每个 token 有独立的 scale，精度最好但管理最复杂

**INT8 vs FP8**：FP8 不需要复杂的 scale 管理，且 H100 原生支持，推荐优先使用 FP8。

### 4.4 更激进的量化：INT4 / INT2 KV Cache

- **KIVI (2024)**：INT2 KV Cache，Key 用 per-channel 量化，Value 用 per-token 量化
- **KVQuant**：混合精度 KV Cache 量化
- 目前仍在研究阶段，尚未在主流框架中广泛采用

---

## 五、KV Cache 复用

### 5.1 Prefix Caching（前缀缓存）

很多推理场景中，不同请求有**相同的前缀**（如系统提示词 system prompt、few-shot examples 等）：

```
请求 1: [System Prompt(1000 tokens)] + "什么是人工智能？"
请求 2: [System Prompt(1000 tokens)] + "解释一下机器学习"

Without Prefix Caching: 每个请求独立计算 System Prompt 的 KV Cache
With Prefix Caching:    只计算一次，三个请求共享 → 节省 2/3 的计算和显存！
```

### 5.2 Automatic Prefix Caching（APC，vLLM）

APC 不需要用户显式指定哪些是前缀，系统自动识别：

```python
# vLLM APC 的核心思想：基于 Block 的 hash
# 每个 physical block 根据其包含的 token 内容计算级联 hash
# Block 0: hash(token_ids[0:16]) = 0xABCD
# Block 1: hash(token_ids[0:16], token_ids[16:32]) = 0x1234（级联 hash）
```

**APC 适用场景**：
- System Prompt 共享：多用户使用相同的 system prompt
- Multi-turn 对话：同一用户连续对话，历史轮次 KV Cache 自动复用
- RAG：多个请求使用相同的检索文档

### 5.3 RadixAttention（SGLang 提出）

RadixAttention 是基于 **Radix Tree（基数树）** 的 KV Cache 复用机制，比 vLLM APC 更精细：

```
KV Cache 组织为 Radix Tree，每个节点对应一段 token 序列的 KV Cache：

                    root
                   /    \
          [System Prompt KV]   [Document A KV]
           /          \              |
[User Question 1 KV] [User Question 2 KV] [Query KV]
```

**APC vs RadixAttention 对比**：

| 特性 | vLLM APC | SGLang RadixAttention |
|------|---------|----------------------|
| 数据结构 | Hash Table（Block 级别） | Radix Tree（Token 级别） |
| 复用粒度 | Block 对齐（16 tokens） | 任意长度 |
| 是否自动 | 需要显式启用 | 默认开启 |
| 淘汰策略 | LRU Block | LRU Node |

---

## 六、KV Cache 迁移（PD 分离架构）

在 Prefill-Decode 分离（PD 分离）架构中，Prefill 节点计算完 KV Cache 后，需要将其迁移到 Decode 节点：

```
┌─────────────────┐                    ┌─────────────────┐
│  Prefill Node   │                    │  Decode Node    │
│  1. 执行 Prefill │                    │                 │
│  2. 生成 KV Cache│   RDMA/TCP 传输    │  5. 接收 KV     │
│  3. 传输 ────── │──────────────────→ │  6. 开始 decode │
│  4. 释放 KV     │                    │                 │
└─────────────────┘                    └─────────────────┘
```

**KV Cache 迁移优化策略**：

1. **流式传输（Layer-by-Layer）**：第 0 层 KV Cache 计算完立即传输，与后续层计算并行
2. **KV Cache 压缩传输**：传输前量化（FP16→FP8），减少传输数据量
3. **GPUDirect RDMA**：直接从 Prefill GPU 显存传输到 Decode GPU 显存，绕过 CPU

---

## 七、KV Cache 淘汰策略

### 7.1 分层淘汰策略

当 GPU 显存不足时，按代价从小到大淘汰：

```
新请求需要 blocks
    ↓
淘汰 Prefix Cache（LRU/LFU）← 代价最小
    ↓（仍不够）
Swap Out 低优先级请求（GPU → CPU）← 代价中等
    ↓（仍不够）
Preempt 低优先级请求（丢弃 KV Cache）← 代价最大（需重新 prefill）
```

### 7.2 常见淘汰算法

- **LRU（最近最少使用）**：淘汰最久没被访问的缓存，简单高效
- **LFU（最少使用频率）**：保留高频使用的缓存（如 system prompt）
- **Size-aware**：优先淘汰大的、价值低的缓存（score = use_count / size）

### 7.3 Swap（GPU ↔ CPU 交换）

```python
# vLLM 的 Swap 实现（概念性）
# Swap Out (GPU → CPU)：将低优先级请求的 KV Cache 移到 CPU 内存
# Swap In (CPU → GPU)：当有空间时，将请求 KV Cache 移回 GPU
# PCIe Gen5: ~64 GB/s，100MB KV Cache 约 1.6ms（比重新 prefill 快得多）
```

---

## 八、面试要点总结

1. **基本概念**：缓存已计算的 K 和 V，将 decode 计算复杂度从 O(N²) 降至 O(N)
2. **显存计算**：`2 × L × H_kv × D × S × dtype_size`，KV Cache 通常占推理显存最大份额
3. **管理策略**：Paged Attention 是主流方案，以 block 为粒度（默认 block_size=16），按需分配
4. **量化**：FP8 KV Cache 是最推荐方案，显存减半，精度影响极小
5. **复用策略**：Prefix Caching（前缀缓存）→ APC（vLLM，基于 Hash）→ RadixAttention（SGLang，基于树，更精细）
6. **迁移**：PD 分离架构中通过 RDMA 传输，可通过 layer-by-layer streaming 与计算重叠
7. **淘汰**：分层淘汰——先 prefix cache（LRU）→ 再 swap to CPU → 最后 preempt（重计算）
8. **注意力机制对 KV Cache 的影响**：MHA > GQA > MQA > MLA（显存占用递减，MLA 最优）

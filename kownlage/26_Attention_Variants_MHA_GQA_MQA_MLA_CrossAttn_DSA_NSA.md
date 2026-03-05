# 6. Attention 变体全面详解：MHA、GQA、MQA、MLA、Cross Attention、DSA、NSA

---

## 6.0 Attention 机制基础回顾

在深入各种变体之前，先回顾标准 Attention 的基本公式：

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

其中：
  Q ∈ R^{n×d_k}    — Query 矩阵
  K ∈ R^{m×d_k}    — Key 矩阵
  V ∈ R^{m×d_v}    — Value 矩阵
  d_k              — Key/Query 维度（用于缩放）
  n                — Query 序列长度
  m                — Key/Value 序列长度（在 Self-Attention 中 n=m）
```

---

## 6.1 MHA（Multi-Head Attention）：标准多头注意力

### 6.1.1 核心思想

MHA 由 Vaswani et al. (2017) 在 "Attention Is All You Need" 中提出。核心思想是：**将注意力计算分成多个独立的"头"（head），每个头在不同的子空间中学习不同的注意力模式，最后拼接并线性变换**。

### 6.1.2 数学公式

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) × W^O

head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)

其中：
  W_i^Q ∈ R^{d_model × d_k}    — 第 i 个头的 Query 投影矩阵
  W_i^K ∈ R^{d_model × d_k}    — 第 i 个头的 Key 投影矩阵
  W_i^V ∈ R^{d_model × d_v}    — 第 i 个头的 Value 投影矩阵
  W^O ∈ R^{h×d_v × d_model}    — 输出投影矩阵
  h                              — 头的数量
  d_k = d_v = d_model / h        — 每个头的维度
```

### 6.1.3 结构示意

```
┌─────────────────────────────────────────────────────┐
│                 Multi-Head Attention                 │
│                                                     │
│  输入: X ∈ R^{n × d_model}                         │
│                                                     │
│  Head 1:  Q1=X·W1^Q   K1=X·W1^K   V1=X·W1^V      │
│           → Attention(Q1, K1, V1) → head_1          │
│                                                     │
│  Head 2:  Q2=X·W2^Q   K2=X·W2^K   V2=X·W2^V      │
│           → Attention(Q2, K2, V2) → head_2          │
│                                                     │
│  ...                                                │
│                                                     │
│  Head h:  Qh=X·Wh^Q   Kh=X·Wh^K   Vh=X·Wh^V      │
│           → Attention(Qh, Kh, Vh) → head_h          │
│                                                     │
│  输出: Concat(head_1, ..., head_h) × W^O            │
└─────────────────────────────────────────────────────┘
```

### 6.1.4 KV Cache 分析

在自回归推理中，每个已生成的 token 的 K 和 V 需要被缓存（避免重复计算）：

```
KV Cache 大小 (每层):
  K cache: batch_size × num_heads × seq_len × d_k
  V cache: batch_size × num_heads × seq_len × d_v

  = 2 × batch_size × num_heads × seq_len × d_head

示例 (Llama-2 70B):
  num_heads = 64, d_head = 128, 80 层
  seq_len = 4096, batch_size = 1, FP16

  每层 KV Cache = 2 × 1 × 64 × 4096 × 128 × 2bytes = 128 MB
  总 KV Cache = 128 MB × 80 = 10.24 GB

  如果 seq_len = 128K:
  总 KV Cache = 10.24 GB × 32 = 327.68 GB！
```

**KV Cache 是长上下文推理的核心瓶颈**。后续的 GQA、MQA、MLA 等变体都在解决这个问题。

### 6.1.5 多头注意力的意义

不同的头可以学习不同类型的注意力模式：
- 有的头关注**局部语法关系**（如相邻词）
- 有的头关注**远距离语义依赖**（如指代消解）
- 有的头关注**位置信息**
- 有的头关注**特定的语言结构**（如主谓关系）

---

## 6.2 MQA（Multi-Query Attention）：多查询注意力

### 6.2.1 核心思想

**论文**: Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)

MQA 的核心改变：**所有 Query 头共享同一组 Key 和 Value**。

```
MHA: 每个头有独立的 Q, K, V
  head_i = Attention(Q_i, K_i, V_i)

MQA: 每个头有独立的 Q，但共享同一个 K 和 V
  head_i = Attention(Q_i, K_shared, V_shared)
```

### 6.2.2 数学公式

```
MultiQuery(Q, K, V) = Concat(head_1, ..., head_h) × W^O

head_i = Attention(Q × W_i^Q, K × W^K, V × W^V)

注意变化：
  W_i^Q ∈ R^{d_model × d_k}     — 每个头有独立的 Q 投影（h 个）
  W^K ∈ R^{d_model × d_k}        — 只有 1 个 K 投影（所有头共享）
  W^V ∈ R^{d_model × d_v}        — 只有 1 个 V 投影（所有头共享）
```

### 6.2.3 结构对比

```
MHA (h=8 头):
  Q: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]   — 8 组独立 Q
  K: [K1, K2, K3, K4, K5, K6, K7, K8]   — 8 组独立 K
  V: [V1, V2, V3, V4, V5, V6, V7, V8]   — 8 组独立 V

MQA (h=8 头):
  Q: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]   — 8 组独立 Q
  K: [K_shared]                           — 1 组共享 K
  V: [V_shared]                           — 1 组共享 V
```

### 6.2.4 KV Cache 优势

```
MHA KV Cache (每层):
  2 × num_heads × seq_len × d_head = 2 × 64 × seq_len × 128

MQA KV Cache (每层):
  2 × 1 × seq_len × d_head = 2 × 1 × seq_len × 128

KV Cache 缩减比: num_heads : 1 = 64 : 1
即减少了 64 倍！
```

### 6.2.5 性能影响

- **推理速度**：大幅提升（内存带宽瓶颈缓解）
- **模型质量**：有一定下降，但 Shazeer 原文报告"只有轻微的质量退化"
- **使用模型**：PaLM、Falcon、StarCoder 等

### 6.2.6 为什么 MQA 能加速推理

自回归解码的瓶颈在于**内存带宽**（memory-bound）而非计算（compute-bound）：

```
Decode 阶段（生成每个新 token）：
  Q: 只有 1 个 token 的 query (很小)
  K, V: 所有历史 token 的 KV Cache (很大)

  计算量: O(seq_len × d)  — 很少
  数据加载量: O(num_kv_heads × seq_len × d) — 很多（需要从 HBM 加载整个 KV Cache）

  计算强度 = FLOPs / 数据量 = 非常低 → 内存带宽瓶颈

  MQA 将 num_kv_heads 从 h 减到 1
  → 需要加载的 KV Cache 数据量减少 h 倍
  → 直接缓解内存带宽瓶颈
  → 推理速度显著提升
```

---

## 6.3 GQA（Grouped Query Attention）：分组查询注意力

### 6.3.1 核心思想

**论文**: Ainslie et al., "GQA: Training Generalized Multi-Query Attention Models are Fast Multi-Head Checkpoints" (2023, EMNLP)

GQA 是 MHA 和 MQA 的**折中方案**：将 Query 头分成 g 组，每组共享一个 KV 头。

```
MHA: num_kv_heads = num_q_heads     (完全独立)
MQA: num_kv_heads = 1               (完全共享)
GQA: num_kv_heads = g               (1 < g < num_q_heads，分组共享)
```

### 6.3.2 数学公式

```
GroupedQuery(Q, K, V) = Concat(head_1, ..., head_h) × W^O

对于第 i 个 Query 头，它使用第 ⌊i × g / h⌋ 个 KV 头:
  group_idx = i × g / h (向下取整)
  head_i = Attention(Q × W_i^Q, K × W_{group_idx}^K, V × W_{group_idx}^V)

其中：
  h = num_query_heads (如 32)
  g = num_kv_heads (如 8)
  每 h/g = 4 个 Query 头共享一组 KV 头
```

### 6.3.3 结构示意

```
GQA (h=8 Query头, g=2 KV组):

  Q 头:  [Q1, Q2, Q3, Q4 | Q5, Q6, Q7, Q8]
          ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓
  KV 组: [  KV_group1    |   KV_group2     ]

  Q1-Q4 共享 KV_group1 (K1, V1)
  Q5-Q8 共享 KV_group2 (K2, V2)

GQA (h=32 Query头, g=8 KV组):  ← Llama-2 70B 使用的配置
  每 4 个 Q 头共享 1 组 KV
```

### 6.3.4 KV Cache 分析

```
GQA KV Cache (每层):
  2 × num_kv_groups × seq_len × d_head

示例 (Llama-2 70B, GQA with g=8):
  每层 = 2 × 8 × 4096 × 128 × 2bytes = 16 MB
  总 = 16 MB × 80 层 = 1.28 GB

对比 MHA (g=64):  10.24 GB → GQA 缩小 8 倍
对比 MQA (g=1):   0.16 GB → GQA 比 MQA 大 8 倍但质量更好
```

### 6.3.5 Uptraining（从 MHA 转换为 GQA）

GQA 论文的一个重要贡献是提出了从已训练好的 MHA 模型转换为 GQA 模型的方法，只需 **5%** 的原始预训练计算量：

```
转换步骤:
1. 将 MHA 的 h 个 KV 头分成 g 组
2. 每组的 KV 权重初始化为组内原始 KV 头的均值
3. 用 5% 的训练数据进行 uptraining

例如: 64 个 KV 头 → 8 组 GQA
  KV_group_1 = mean(K_1, K_2, ..., K_8)  (原始头 1-8 的均值)
  KV_group_2 = mean(K_9, K_10, ..., K_16) (原始头 9-16 的均值)
  ...
```

### 6.3.6 使用 GQA 的模型

| 模型 | num_q_heads | num_kv_heads | 头比率 |
|------|------------|-------------|--------|
| Llama-2 70B | 64 | 8 | 8:1 |
| Llama-3 8B | 32 | 8 | 4:1 |
| Llama-3 70B | 64 | 8 | 8:1 |
| Mistral 7B | 32 | 8 | 4:1 |
| Gemma | 16 | 1 | 16:1 (实际是 MQA) |
| Qwen-2 72B | 64 | 8 | 8:1 |

---

## 6.4 MLA（Multi-head Latent Attention）：多头潜在注意力

### 6.4.1 核心思想

**来源**: DeepSeek-V2 (2024)

MLA 的核心思想与 GQA/MQA 不同。GQA/MQA 通过**减少 KV 头数**来减少 KV Cache，而 MLA 通过**低秩压缩**将 KV 投影到一个低维潜在空间，从而大幅减少 KV Cache。

```
GQA/MQA 的思路: 减少 KV 头的数量
MLA 的思路:     压缩每个 token 的 KV 表示维度
```

### 6.4.2 数学公式

#### 标准 MHA 的 KV 计算

```
对于每个 token 的隐藏状态 h_t ∈ R^{d_model}:

  K_t = h_t × W^K    (W^K ∈ R^{d_model × d_k·n_h})
  V_t = h_t × W^V    (W^V ∈ R^{d_model × d_v·n_h})

KV Cache 存储: [K_t, V_t] ∈ R^{2·n_h·d_k}  (每个 token)
```

#### MLA 的低秩压缩

```
Step 1: 下投影（压缩）
  c_t^{KV} = h_t × W^{DKV}    (W^{DKV} ∈ R^{d_model × d_c})

  其中 d_c << n_h × d_k（如 d_c = 512, 而 n_h × d_k = 128 × 128 = 16384）

KV Cache 只需存储: c_t^{KV} ∈ R^{d_c}  (每个 token)

Step 2: 上投影（解压，推理时计算）
  K_t = c_t^{KV} × W^{UK}     (W^{UK} ∈ R^{d_c × d_k·n_h})
  V_t = c_t^{KV} × W^{UV}     (W^{UV} ∈ R^{d_c × d_v·n_h})
```

**关键洞察**：K 和 V 共享同一个低秩 latent 向量 c_t^{KV}，在 KV Cache 中只需存储这个向量。

### 6.4.3 推理时的矩阵吸收优化

MLA 的另一个关键优化是**矩阵吸收（Matrix Absorption）**：

```
标准 Attention 计算:
  Attention_score = Q_t × K_s^T = (h_t × W^Q) × (c_s^{KV} × W^{UK})^T
                  = h_t × W^Q × (W^{UK})^T × (c_s^{KV})^T

矩阵吸收:
  令 W^{Q'} = W^Q × (W^{UK})^T   (离线预计算)
  则 Attention_score = h_t × W^{Q'} × (c_s^{KV})^T

  Q' = h_t × W^{Q'}  可以直接与 c_s^{KV} 做点积
  → 不需要显式计算 K_s = c_s^{KV} × W^{UK}

同理，输出计算:
  output = Attention_weights × V = Attention_weights × (c_s^{KV} × W^{UV})
         = (Attention_weights × c_s^{KV}) × W^{UV}

  可以进一步将 W^{UV} 吸收到输出投影 W^O 中:
  令 W^{O'} = W^{UV} × W^O  (离线预计算)
  则 output = (Attention_weights × c_s^{KV}) × W^{O'}
```

**这意味着推理时：**
1. KV Cache 只存 c_t^{KV}（极小）
2. 不需要将 c_t^{KV} 解压回完整的 K、V（W^{UK} 和 W^{UV} 被吸收了）
3. 直接用压缩后的 latent 向量做 Attention 计算

### 6.4.4 RoPE 的处理

位置编码（RoPE）在 MLA 中需要特殊处理，因为 RoPE 需要作用在 K 上，但 K 被压缩了：

```
问题: RoPE(K) = RoPE(c^{KV} × W^{UK})
      → 这让 W^{UK} 无法被吸收（因为 RoPE 在中间）

解决方案: 引入额外的 RoPE Key
  K_t = [K_t^{content}, K_t^{rope}]

  K_t^{content} = c_t^{KV} × W^{UK}    — 内容相关（可吸收，不加 RoPE）
  K_t^{rope} = h_t × W^{KR}            — 位置相关（加 RoPE，需要额外缓存）

  最终: K_t = Concat(K_t^{content}, RoPE(K_t^{rope}))

KV Cache 存储:
  c_t^{KV} ∈ R^{d_c}                    — 压缩后的 latent 向量
  K_t^{rope} ∈ R^{d_rope}               — RoPE key（维度较小）

总 KV Cache = d_c + d_rope（远小于原始 n_h × d_k）
```

### 6.4.5 KV Cache 对比

```
DeepSeek-V2 配置:
  d_model = 5120, n_h = 128, d_k = 128
  d_c = 512 (latent 维度)
  d_rope = 64 (RoPE key 维度)

标准 MHA KV Cache 每 token 每层:
  2 × n_h × d_k = 2 × 128 × 128 = 32768 维度

MLA KV Cache 每 token 每层:
  d_c + d_rope = 512 + 64 = 576 维度

压缩比: 32768 / 576 ≈ 56.9 倍

论文报告: KV Cache 减少 93.3%（1 - 576/32768 × 某些其他因素 ≈ 93.3%）
```

### 6.4.6 MLA 的优势总结

1. **极致的 KV Cache 压缩**：不是减少 KV 头（像 GQA），而是压缩每个 token 的 KV 表示
2. **保持模型容量**：Query 侧仍然是完整的多头，保留了丰富的表达能力
3. **矩阵吸收优化**：推理时不需要解压 KV，直接在 latent 空间计算
4. **比 GQA 更灵活**：不需要限制 KV 头数，压缩率可以通过 d_c 连续调节

---

## 6.5 Cross Attention：交叉注意力

### 6.5.1 核心思想

Cross Attention 是 Encoder-Decoder 架构中的核心组件。与 Self-Attention 不同，Cross Attention 的 Q 来自一个序列（通常是 Decoder），而 K 和 V 来自另一个序列（通常是 Encoder）。

### 6.5.2 数学公式

```
CrossAttention(Q_decoder, K_encoder, V_encoder)
  = softmax(Q_decoder × K_encoder^T / √d_k) × V_encoder

其中:
  Q_decoder = Decoder_hidden_state × W^Q     — Query 来自 Decoder
  K_encoder = Encoder_output × W^K            — Key 来自 Encoder
  V_encoder = Encoder_output × W^V            — Value 来自 Encoder
```

### 6.5.3 结构示意

```
┌──────────────────────────────────────────┐
│            Cross Attention                │
│                                          │
│   Encoder Output                         │
│   ┌──────────┐                           │
│   │  token_1  │──→ K1, V1               │
│   │  token_2  │──→ K2, V2               │
│   │  ...      │──→ ...                  │
│   │  token_m  │──→ Km, Vm               │
│   └──────────┘                           │
│                                          │
│   Decoder Hidden State                   │
│   ┌──────────┐                           │
│   │  token_t  │──→ Q_t                  │
│   └──────────┘                           │
│                                          │
│   Q_t 与所有 K1..Km 计算注意力           │
│   → 加权求和 V1..Vm                      │
│   → 输出: Decoder token_t 对              │
│         Encoder 所有 token 的注意力表示    │
└──────────────────────────────────────────┘
```

### 6.5.4 应用场景

| 场景 | Q 来源 | K/V 来源 | 模型示例 |
|------|--------|---------|---------|
| 机器翻译 | 目标语言 Decoder | 源语言 Encoder | T5, mBART |
| 语音识别 | 文本 Decoder | 音频 Encoder | Whisper |
| 图文理解 | 文本 Decoder | 视觉 Encoder | Flamingo, LLaVA |
| 文本摘要 | 摘要 Decoder | 原文 Encoder | BART, Pegasus |
| 扩散模型 | UNet 特征 | 文本 Embedding | Stable Diffusion |

### 6.5.5 Cross Attention 的 KV Cache 特性

```
Cross Attention 的 KV Cache 有一个独特优势:
  Encoder 输出在整个生成过程中是固定的
  → K_encoder 和 V_encoder 只需计算一次
  → KV Cache 不会随着生成长度增长
  → 比 Self-Attention 的 KV Cache 高效得多
```

### 6.5.6 现代多模态模型中的 Cross Attention

在现代多模态大模型中，Cross Attention 被广泛用于将视觉信息注入文本模型：

```
典型的视觉-语言模型架构:

方案 A: 显式 Cross Attention (如 Flamingo)
  Decoder Layer:
    Self-Attention → Cross-Attention(text→image) → FFN

方案 B: 隐式 Cross Attention (如 LLaVA)
  将图像特征转化为 "visual tokens"，拼接到文本 token 序列中
  Self-Attention 自然实现了文本对图像的注意力
  （本质是将 Cross Attention 转化为 Self-Attention 的一部分）
```

---

## 6.6 DSA（DeepSeek Attention）

### 6.6.1 什么是 DSA

DSA 并非一个独立的标准术语，而是对 DeepSeek 系列模型中使用的注意力机制的统称。在 DeepSeek-V2/V3 中，核心注意力机制就是 **MLA（Multi-head Latent Attention）**。

因此 DSA 的具体机制实际上就是上面 6.4 节中详述的 MLA。但是 DeepSeek 在不同版本中对注意力机制有不同的改进，可以概括如下：

### 6.6.2 DeepSeek 注意力机制的演进

```
DeepSeek-V1 (2024.01):
  → 标准 MHA + RoPE
  → 无特殊优化

DeepSeek-V2 (2024.05):
  → MLA (Multi-head Latent Attention)
  → 低秩 KV 压缩 + 矩阵吸收
  → KV Cache 减少 93.3%

DeepSeek-V3 (2024.12):
  → MLA (继承自 V2，经过验证)
  → 配合 Multi-Token Prediction
  → 推理优化更深入

DeepSeek-R1 (2025.01):
  → 基于 V3 的 MLA
  → 推理链（Chain of Thought）对长上下文的需求更高
  → MLA 的 KV Cache 压缩优势更加突出
```

### 6.6.3 DSA/MLA 的独特设计哲学

与 GQA/MQA 的对比思考：

```
GQA 的设计哲学:
  "让多个 Q 头共享一个 KV 头"
  → 在模型结构上做减法
  → 简单直接但牺牲了 KV 的表达能力
  → 不同的 KV 组之间不共享信息

MLA 的设计哲学:
  "将 KV 投影到低维空间，从低维空间重建"
  → 在信息压缩上做文章
  → 保留了完整多头注意力的表达能力
  → 通过低秩分解实现信息的紧凑表示
  → 所有 KV 头共享同一个 latent 向量（信息共享更充分）
```

### 6.6.4 MLA 在推理框架中的支持

MLA 的特殊结构需要推理框架的专门支持：

```
FlashInfer:    原生支持 MLA Attention（有专门优化的内核）
SGLang:        支持 MLA（通过 FlashInfer 后端）
vLLM:          支持 MLA（通过 FlashInfer 或自实现的内核）
TensorRT-LLM:  支持 MLA（NVIDIA 官方适配）
```

---

## 6.7 NSA（Native Sparse Attention）：原生稀疏注意力

### 6.7.1 核心思想

**论文**: DeepSeek, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (2025.02)

NSA 的核心思想：**不是所有 token 对当前 token 的贡献都同等重要。通过动态选择最相关的 token 子集来计算注意力，可以在保持质量的同时大幅减少计算量。**

与之前的稀疏注意力方法不同，NSA 是**原生可训练的**（在预训练阶段就使用稀疏注意力）和**硬件对齐的**（算法设计考虑了 GPU 的硬件特性）。

### 6.7.2 三分支架构

NSA 使用**三个互补的分支**来处理不同范围的上下文：

```
┌──────────────────────────────────────────────────────────┐
│                     NSA Architecture                      │
│                                                          │
│  Input: Q_t, KV_cache = {K_1, V_1, ..., K_{t-1}, V_{t-1}} │
│                                                          │
│  ┌──────────────────┐                                    │
│  │  Branch 1:        │                                    │
│  │  Token Compression│  粗粒度全局感知                      │
│  │  (压缩分支)        │  将 token 块压缩为摘要              │
│  └────────┬─────────┘                                    │
│           │                                              │
│  ┌────────┴─────────┐                                    │
│  │  Branch 2:        │                                    │
│  │  Token Selection  │  细粒度精确选择                      │
│  │  (选择分支)        │  选择最相关的 token 块              │
│  └────────┬─────────┘                                    │
│           │                                              │
│  ┌────────┴─────────┐                                    │
│  │  Branch 3:        │                                    │
│  │  Sliding Window   │  局部上下文                         │
│  │  (滑动窗口分支)    │  关注最近的 token                   │
│  └────────┬─────────┘                                    │
│           │                                              │
│  ┌────────┴─────────┐                                    │
│  │   Gating Network  │  动态加权三个分支的输出              │
│  └──────────────────┘                                    │
│                                                          │
│  Output = g_1 × Compressed_Attn + g_2 × Selected_Attn    │
│         + g_3 × Window_Attn                              │
└──────────────────────────────────────────────────────────┘
```

#### Branch 1: Token Compression（令牌压缩分支）

```
功能: 提供全局上下文感知

步骤:
1. 将历史 KV Cache 分成连续的块（block），每块 block_l 个 token
2. 每个块通过压缩函数 φ 压缩为一个代表性向量
3. 用 Q 对压缩后的向量做标准 Attention

压缩函数 φ 的实现:
  将 block_l 个 token 的表示进行层级下采样
  例如 block_l = 32：
    32 tokens → 16 pairs → 8 → 4 → 2 → 1 个压缩表示
    每步通过线性层 + SiLU 激活函数融合

优势: 以 O(N/block_l) 的代价获得全局信息概览
```

#### Branch 2: Token Selection（令牌选择分支）

```
功能: 精确选择最相关的 token 块

步骤:
1. 将历史 KV 分成块
2. 计算 Q 与每个块的"重要性分数"
3. 选择得分最高的 Top-K 个块
4. 只对选中的块做完整的 Attention

选择策略:
  score_j = Q_t × mean(K_block_j)^T  或其他快速估计方法
  selected_blocks = TopK(scores)

优势: 以 O(K × block_size) 的代价获得精确的细粒度信息
```

#### Branch 3: Sliding Window（滑动窗口分支）

```
功能: 捕获局部上下文信息

实现: 标准滑动窗口注意力
  只关注最近的 W 个 token

优势: O(W) 的代价确保局部信息不丢失（W 通常为 512-1024）
```

### 6.7.3 门控机制

三个分支通过可学习的门控网络动态组合：

```python
# 门控网络
class NSAGating(nn.Module):
    def __init__(self, hidden_dim):
        self.gate = nn.Linear(hidden_dim, 3)  # 3 个分支

    def forward(self, x):
        gates = torch.sigmoid(self.gate(x))   # [batch, seq, 3]
        g_compress = gates[..., 0:1]
        g_select = gates[..., 1:2]
        g_window = gates[..., 2:3]
        return g_compress, g_select, g_window

# NSA 输出
output = (g_compress * compress_attn_output +
          g_select * select_attn_output +
          g_window * window_attn_output)
```

**门控的动态性**：
- 对于需要全局理解的 query（如总结性问题），压缩分支权重高
- 对于需要精确信息的 query（如事实问题），选择分支权重高
- 对于需要局部上下文的 query（如代词消解），滑动窗口权重高

### 6.7.4 硬件对齐设计

NSA 特别强调与 GPU 硬件特性的对齐：

```
1. 块级操作（Block-wise Operations）:
   所有操作都基于 token 块而非单个 token
   → 块大小对齐 GPU 的 memory transaction 粒度
   → 更好的内存合并访问（coalesced memory access）

2. 算术强度平衡（Arithmetic Intensity Balanced）:
   精心设计块大小和选择数量
   → 确保计算/内存访问比率处于 GPU 的"甜蜜点"
   → 避免纯 memory-bound 或纯 compute-bound

3. 与 Tensor Core 的兼容:
   矩阵乘法维度对齐 Tensor Core 的要求（通常是 16 的倍数）
```

### 6.7.5 NSA 的训练优势

与许多稀疏注意力方法不同，NSA 是**原生可训练**的：

```
传统稀疏注意力:
  训练时用 Full Attention
  推理时切换到 Sparse Attention（可能不匹配）
  → 训练和推理的不一致性可能导致性能下降

NSA:
  训练时就使用 Sparse Attention
  推理时使用同样的 Sparse Attention
  → 训练和推理完全一致
  → 模型学会了如何最有效地利用稀疏注意力模式
  → 减少了预训练计算量（因为训练时也是稀疏的）
```

### 6.7.6 性能

```
64K 序列长度下的加速:
  相比 Full Attention:
    Decoding: 显著加速（减少了需要访问的 KV Cache 量）
    Forward: 显著加速（减少了注意力计算量）
    Backward: 显著加速（减少了梯度计算量）

质量:
  通用基准测试: 持平或超过 Full Attention
  长上下文任务: 持平或超过 Full Attention
  指令跟随推理: 持平或超过 Full Attention
```

---

## 6.8 各种 Attention 变体对 KV Cache 大小的全面对比

### 6.8.1 定量对比

假设模型配置：
```
d_model = 8192
num_q_heads (h) = 64
d_head = 128 (d_model / h)
num_layers = 80
seq_len = 4096
batch_size = 1
精度: FP16 (2 bytes)
```

| 方法 | KV Cache 每 token 每层 (维度数) | KV Cache 每 token 每层 (bytes) | 总 KV Cache (seq=4K) | 相对 MHA |
|------|-------------------------------|-------------------------------|---------------------|---------|
| **MHA** | 2 × 64 × 128 = 16384 | 32,768 B | 10.24 GB | 1.00× |
| **GQA** (g=8) | 2 × 8 × 128 = 2048 | 4,096 B | 1.28 GB | 0.125× |
| **MQA** (g=1) | 2 × 1 × 128 = 256 | 512 B | 0.16 GB | 0.016× |
| **MLA** (d_c=512, d_r=64) | 512 + 64 = 576 | 1,152 B | 0.36 GB | 0.035× |

### 6.8.2 可视化对比

```
KV Cache 大小 (相对于 MHA = 100%):

MHA:            ████████████████████████████████████████ 100%
GQA (g=8):      █████ 12.5%
MLA (d_c=512):  ██ 3.5%
MQA (g=1):      █ 1.6%
```

### 6.8.3 KV Cache vs 模型质量的权衡

```
                    模型质量
                    ↑
              MHA ●──────────────────── 最高质量，最大 KV Cache
                  │
            MLA ● │──────────────── 接近 MHA 质量，极小 KV Cache（最佳权衡点）
                  │
          GQA ●   │──────────── 接近 MHA 质量，中等 KV Cache
                  │
        MQA ●     │──────── 略低质量，极小 KV Cache
                  │
                  └──────────────────────→ KV Cache 大小
```

### 6.8.4 长上下文场景下的影响

```
seq_len = 128K 时的 KV Cache（80 层，batch=1，FP16）:

MHA:  2 × 64 × 128 × 128K × 80 × 2 = 327.68 GB
      → 需要 5+ 个 80GB GPU 仅用于 KV Cache！

GQA:  2 × 8 × 128 × 128K × 80 × 2 = 40.96 GB
      → 1 个 80GB GPU 可能够用

MLA:  576 × 128K × 80 × 2 = 11.53 GB
      → 轻松放入单 GPU

MQA:  2 × 128 × 128K × 80 × 2 = 5.12 GB
      → 非常小
```

这解释了为什么 DeepSeek-V2 选择 MLA：在 128K 上下文长度下，MLA 让 KV Cache 从不可行变为完全可行。

### 6.8.5 不同 Attention 变体的适用场景

| 变体 | 适用场景 | 代表模型 |
|------|---------|---------|
| **MHA** | 模型质量优先，上下文不太长 | GPT-3, BERT, 早期 LLM |
| **GQA** | 质量和效率的平衡 | Llama-2/3, Mistral, Qwen-2 |
| **MQA** | 极致推理速度，质量要求不极端 | PaLM, Falcon, StarCoder |
| **MLA** | 超长上下文 + 高质量 | DeepSeek-V2/V3/R1 |
| **Cross Attention** | 多模态/Encoder-Decoder | T5, Whisper, Stable Diffusion |
| **NSA** | 超长序列 + 训练加速 | DeepSeek 未来模型 |

---

## 6.9 总结表

| 特性 | MHA | MQA | GQA | MLA | Cross Attn | NSA |
|------|-----|-----|-----|-----|-----------|-----|
| Q 头数 | h | h | h | h | h | h |
| KV 头数 | h | 1 | g | h(低秩) | h | h(稀疏) |
| KV Cache 大小 | 最大 | 最小 | 中等 | 很小 | 固定 | 减少(稀疏) |
| 模型质量 | 最高 | 较低 | 接近MHA | 接近MHA | - | 接近MHA |
| 推理速度 | 最慢 | 最快 | 较快 | 快 | - | 很快 |
| 训练兼容 | 是 | 是 | 是 | 是 | 是 | 是(原生) |
| 实现复杂度 | 低 | 低 | 低 | 高 | 低 | 高 |
| 适用上下文 | 短-中 | 短-中 | 中-长 | 超长 | 取决于编码器 | 超长 |

# 24. Transformer 权重的含义与作用

---

## 一、Transformer 架构总览

现代 LLM 的 Transformer 架构（以 Llama 为例）由以下组件构成：

```
输入 Token IDs
    ↓
Token Embedding (embed_tokens)
    ↓
[重复 N 次]
┌─────────────────────────────────────────────────────┐
│  Transformer Layer i                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │  RMSNorm (input_layernorm)                   │   │
│  │      ↓                                       │   │
│  │  Self-Attention:                             │   │
│  │    Q_proj, K_proj, V_proj, O_proj            │   │
│  └──────────────────────────────────────────────┘   │
│           ↓ (残差连接)                               │
│  ┌──────────────────────────────────────────────┐   │
│  │  RMSNorm (post_attention_layernorm)          │   │
│  │      ↓                                       │   │
│  │  FFN (SwiGLU):                               │   │
│  │    gate_proj, up_proj, down_proj             │   │
│  └──────────────────────────────────────────────┘   │
│           ↓ (残差连接)                               │
└─────────────────────────────────────────────────────┘
    ↓
RMSNorm (norm)
    ↓
LM Head (lm_head)
```

---

## 二、Embedding 权重

### 2.1 embed_tokens（词嵌入）

- **形状**：`[vocab_size, hidden_dim]`
  - 例如 Llama-2-7B：`[32000, 4096]`
  - 参数量：32000 × 4096 = 131M 参数

- **作用**：将 Token ID 映射为 hidden_dim 维的稠密向量
- **本质**：一个可训练的查找表（Lookup Table），第 i 行是 Token i 的向量表示
- **初始化**：通常使用随机初始化，训练过程中学习每个 token 的语义表示

**使用方式**：
```python
# 输入：token_ids = [101, 2054, ...]
# 输出：hidden_states = embed_tokens[token_ids]  # 形状 [batch, seq_len, hidden_dim]
```

### 2.2 lm_head（语言模型输出头）

- **形状**：`[vocab_size, hidden_dim]`（与 embed_tokens 形状相同）
- **作用**：将最后一层的 hidden state 映射到 vocab_size 维的 logits
- **权重共享（Weight Tying）**：很多模型中 `lm_head.weight = embed_tokens.weight`（权重共享），减少参数量

```python
# 输出 logits
logits = hidden_states @ lm_head.weight.T  # [batch, seq_len, vocab_size]
probs = softmax(logits[:, -1, :])  # 最后一个 token 的概率分布
next_token = sample(probs)
```

---

## 三、Normalization 权重

### 3.1 RMSNorm（均方根归一化）

Llama 系列使用 RMSNorm 代替 LayerNorm：

**公式**：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

- **权重（$\gamma$，也叫 weight）形状**：`[hidden_dim]`
  - 例如 Llama-2-7B：`[4096]`
  - 参数量极少，但对稳定训练很重要

- **作用**：
  - RMSNorm 对 hidden state 做归一化，防止梯度爆炸/消失
  - $\gamma$ 是可学习的缩放参数，允许模型恢复表示能力

- **位置**：
  - `input_layernorm`：Attention 层前的归一化（Pre-Norm 架构）
  - `post_attention_layernorm`：FFN 层前的归一化
  - `norm`（model.norm）：最后一层 Transformer 后，LM Head 前

**Pre-Norm vs Post-Norm**：
- Pre-Norm（Llama 等）：归一化在子层前 → 训练更稳定
- Post-Norm（原始 Transformer）：归一化在子层后 → 理论表达能力更强

---

## 四、Self-Attention 权重

### 4.1 Q_proj（查询投影矩阵）

- **形状**：`[num_heads × head_dim, hidden_dim]` = `[hidden_dim, hidden_dim]`（MHA）
  - 例如 Llama-2-7B：`[4096, 4096]`（32 heads × 128 head_dim）
  - GQA 时（如 Llama-3-70B）：`[num_kv_heads × head_dim, hidden_dim]`

- **作用**：将输入 hidden state 投影为 Query 向量
  - $Q = XW_Q$，形状 `[seq_len, num_heads × head_dim]`
  - Reshape 为 `[batch, num_heads, seq_len, head_dim]`

### 4.2 K_proj（键投影矩阵）

- **形状**（GQA）：`[num_kv_heads × head_dim, hidden_dim]`
  - Llama-3-8B（GQA, 8 KV heads）：`[1024, 4096]`（8 × 128）
  - 比 Q_proj 小（GQA 设计，减少 KV Cache）

- **作用**：将 hidden state 投影为 Key 向量（存入 KV Cache）
  - $K = XW_K$

### 4.3 V_proj（值投影矩阵）

- **形状**：与 K_proj 相同（GQA 时）
- **作用**：将 hidden state 投影为 Value 向量（存入 KV Cache）
  - $V = XW_V$

### 4.4 O_proj（输出投影矩阵）

- **形状**：`[hidden_dim, num_heads × head_dim]` = `[hidden_dim, hidden_dim]`
  - 例如 Llama-2-7B：`[4096, 4096]`

- **作用**：将多头 Attention 的输出拼接后投影回 hidden_dim
  - $\text{Output} = \text{concat}(\text{head}_1, ..., \text{head}_h) \cdot W_O$

**Attention 完整计算**：
```
Q = X @ W_Q.T       → [batch, seq, num_heads, head_dim]
K = X @ W_K.T       → [batch, seq, num_kv_heads, head_dim]
V = X @ W_V.T       → [batch, seq, num_kv_heads, head_dim]

# 缩放点积 Attention
scores = Q @ K.T / sqrt(head_dim)   → [batch, num_heads, seq, seq]
weights = softmax(scores + causal_mask)
attn_out = weights @ V              → [batch, num_heads, seq, head_dim]

# 输出投影
output = reshape(attn_out) @ W_O.T  → [batch, seq, hidden_dim]
```

---

## 五、FFN 权重（SwiGLU）

现代 LLM（Llama、DeepSeek 等）使用 SwiGLU 激活函数的 FFN：

$$\text{FFN}(x) = \text{SiLU}(xW_\text{gate}) \odot (xW_\text{up}) \cdot W_\text{down}$$

### 5.1 gate_proj

- **形状**：`[intermediate_size, hidden_dim]`
  - Llama-2-7B：`[11008, 4096]`（intermediate_size ≈ 2.67 × hidden_dim）
  - 参数量：11008 × 4096 ≈ 45M

- **作用**：计算 SiLU 的门控信号
  - `gate = SiLU(X @ W_gate.T)`

### 5.2 up_proj

- **形状**：`[intermediate_size, hidden_dim]`（与 gate_proj 相同）

- **作用**：计算 FFN 的"主干"向量
  - `up = X @ W_up.T`

### 5.3 down_proj

- **形状**：`[hidden_dim, intermediate_size]`（gate_proj 的转置形状）
  - Llama-2-7B：`[4096, 11008]`

- **作用**：将中间表示投影回 hidden_dim
  - `output = (gate * up) @ W_down.T`

**FFN 完整计算**：
```python
gate = F.silu(x @ gate_proj.T)    # [batch, seq, intermediate_size]
up = x @ up_proj.T                # [batch, seq, intermediate_size]
output = (gate * up) @ down_proj.T  # [batch, seq, hidden_dim]
```

**为什么用 SwiGLU？**
- 比 ReLU FFN 更强的表达能力
- SiLU（Sigmoid Linear Unit）: `SiLU(x) = x * sigmoid(x)` 是平滑的非线性激活
- 门控机制让 FFN 可以"选择性"地放大/抑制信息

---

## 六、MoE 模型中的 Expert 权重

对于 MoE 模型（如 DeepSeek V3），FFN 被替换为多个 Expert FFN：

### 6.1 Router（路由器）

- **形状**：`[num_experts, hidden_dim]`
  - DeepSeek V3：256 个 Expert，8 个被激活
  - `[256, 7168]`

- **作用**：计算每个 token 应该去哪个 Expert
  ```python
  scores = x @ router.T   # [batch, seq, num_experts]
  topk_indices = topk(scores, k=8)  # 选 top-8 Expert
  ```

### 6.2 Expert FFN

- 每个 Expert 有独立的 gate_proj、up_proj、down_proj 权重
- 参数量 = num_experts × (单个 Expert 参数量)
- 每次推理只激活 top-K 个 Expert，其他 Expert 不参与计算

### 6.3 Shared Expert（共享专家）

DeepSeek MoE 设计中有一类 Shared Expert，所有 token 都会通过这些 Expert（类似 Dense FFN），与可路由的 Expert 并行计算。

---

## 七、不同模型架构的权重差异

| 架构 | 特点 | 代表模型 |
|------|------|---------|
| MHA（标准多头注意力） | Q/K/V/O 各一个矩阵 | GPT-2, BERT |
| GQA（分组查询注意力） | K/V 头数 < Q 头数，减少 KV Cache | Llama-3, Mistral |
| MQA（多查询注意力） | 只有 1 个 K/V 头 | PaLM, Falcon |
| MLA（多头潜在注意力）| K/V 压缩为低维潜在空间 | DeepSeek V2/V3 |
| SwiGLU FFN | gate × up → down | Llama-2/3, DeepSeek |
| Dense FFN | W1 → W2 | GPT-2（GELU） |
| MoE FFN | 多 Expert + Router | Mixtral, DeepSeek V3 |

---

## 八、权重的显存占用

以 Llama-2-7B 为例（BF16，每个参数 2 字节）：

| 模块 | 参数量 | 显存 |
|------|--------|------|
| embed_tokens | 32000 × 4096 = 131M | 262 MB |
| 每层 Attention（32层） | Q+K+V+O = 4 × 4096² = 67M | 134 MB/层 |
| 每层 FFN（32层） | gate+up+down = 3 × 4096 × 11008 = 135M | 270 MB/层 |
| 每层 Norm（×2）| 2 × 4096 ≈ 0 | 可忽略 |
| lm_head（共享）| 0（共享 embed_tokens） | 0 |
| **总计** | ~6.7B 参数 | **~13.4 GB** |

---

## 九、总结

理解每个权重的含义有助于：
1. **调试**：定位 NaN/Inf，确认各层激活值
2. **量化**：针对不同权重选择不同量化策略（如 lm_head 不量化）
3. **LoRA**：确定对哪些权重施加 LoRA（通常是 Q/K/V/O 和 FFN）
4. **推理优化**：理解哪些权重是计算瓶颈，针对性优化
5. **显存估算**：预测模型在目标硬件上能否部署

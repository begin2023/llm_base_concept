# 27. Transformer 位置编码（RoPE、ALiBi）

---

## 一、为什么需要位置编码

Transformer 的 Self-Attention 本质上是对序列中所有 token 做加权求和，这个操作是**排列不变的（permutation-invariant）**——如果打乱输入序列的顺序，Attention 的输出（忽略 causal mask）不会改变。

但语言是有顺序的，"猫吃鱼"和"鱼吃猫"意思完全不同。因此必须引入位置信息。

---

## 二、绝对位置编码（已较少使用）

### 2.1 Sinusoidal 位置编码（原始 Transformer）

原始 Transformer 论文（Vaswani et al. 2017）使用固定的正弦/余弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- pos：token 在序列中的位置
- i：embedding 维度的索引
- d：embedding 维度

**特点**：
- 固定不可学习
- 不同位置的编码相互正交
- 理论上支持任意长度（外推性差）

### 2.2 可学习的绝对位置编码（BERT、GPT-2）

$$\text{PE}_\text{learned} \in \mathbb{R}^{\text{max\_seq\_len} \times d}$$

训练时随机初始化，学习每个位置的向量。

**问题**：
- 无法外推（超出训练时的最大长度会崩溃）
- 不同位置之间没有显式的"相对远近"关系

---

## 三、RoPE（Rotary Position Embedding，旋转位置编码）

RoPE 是目前最主流的位置编码方法，被 Llama、Mistral、Qwen、DeepSeek 等几乎所有主流模型采用。

### 3.1 核心思想

RoPE 不是在 embedding 上加位置信息，而是在 Attention 的 Q/K 计算时，**将位置信息编码为向量的旋转**。

**数学原理**（2D 情况）：

对于位置 m 处的查询向量 $q = [q_1, q_2]^T$，RoPE 将其旋转角度 $m\theta$：

$$\text{RoPE}(q, m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix}$$

**高维情况**：将 head_dim 维向量分成 head_dim/2 对，每对独立旋转，旋转频率不同：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2-1$$

### 3.2 关键性质

**相对位置的内积只依赖相对距离**：

$$\langle \text{RoPE}(q_m), \text{RoPE}(k_n) \rangle = f(q, k, m-n)$$

这意味着位置 m 的 Q 与位置 n 的 K 的点积，只与相对距离 (m-n) 有关，不依赖绝对位置——这是 RoPE 设计的核心数学优美之处。

### 3.3 RoPE 的实现

```python
import torch

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 cos/sin 值"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [seq_len, head_dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """应用 RoPE"""
    # xq: [batch, seq_len, num_heads, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[:xq_.shape[1]]  # 截取当前序列长度

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 3.4 RoPE 的优缺点

**优点**：
- 天然支持相对位置关系
- 对序列长度外推有一定能力（相比绝对位置编码）
- 实现简单，计算高效
- 与 FlashAttention 等优化兼容好

**缺点**：
- 训练长度以外的位置外推性能下降（虽比绝对位置编码好，但有限）
- 需要进行长度外推时需要额外技术（见第 33 章）

---

## 四、ALiBi（Attention with Linear Biases）

ALiBi 是另一种位置编码方案，由 Press et al. 2022 提出，被 BLOOM、MPT 等模型采用。

### 4.1 核心思想

ALiBi 不修改 Q/K/V 的计算，而是在 Attention score 上**直接添加线性偏置（Bias）**：

$$\text{Attention Score}_{ij} = q_i k_j^T - m \cdot |i - j|$$

- $|i - j|$：Query 位置 i 和 Key 位置 j 的距离
- $m$：每个 attention head 独有的斜率（slope），不可学习，按固定规则设置

**斜率的设置**：
- 对于 n 个 attention head，斜率集合为 $\{m_1, m_2, ..., m_n\}$
- 按照等比数列：$m_i = 2^{-8i/n}$（对于 8 头，斜率约为 0.5, 0.25, 0.125, ...）

### 4.2 ALiBi 的效果

```
位置距离 →
距离 0: 偏置 = 0
距离 1: 偏置 = -m
距离 2: 偏置 = -2m
距离 k: 偏置 = -km
```

效果：距离越远的 token，其 Attention score 被惩罚越多，模型天然偏向关注近距离 token。

### 4.3 ALiBi 的外推优势

ALiBi 的核心优势：**零样本外推（Zero-shot Extrapolation）**

- 训练时使用 2048 token
- 推理时可以直接处理 4096、8192 甚至更长的序列，几乎不需要额外调整
- 这是因为 ALiBi 的"线性衰减"特性在任意长度下都成立

### 4.4 ALiBi vs RoPE 对比

| 特性 | RoPE | ALiBi |
|------|------|-------|
| 作用位置 | Q/K 向量（旋转） | Attention Score（加偏置） |
| 长度外推 | 需要额外技术（NTK、YaRN 等） | 天然支持，零样本外推 |
| 下游任务性能 | 更强（大多数任务） | 略弱（但外推性更好） |
| 主流模型 | Llama、Qwen、DeepSeek | BLOOM、MPT |
| 计算开销 | 略高（旋转操作） | 低（加法） |

---

## 五、其他位置编码方法

### 5.1 T5 的相对位置编码

T5 模型使用可学习的相对位置偏置：
- 将相对距离分桶（bucket），每个桶对应一个可学习的偏置值
- 添加到 Attention score 上
- 支持一定程度的外推

### 5.2 KERPLE

KERPLE 将 ALiBi 推广，使用更灵活的衰减函数（指数衰减而非线性衰减）。

### 5.3 Sandwich Encoding

将绝对位置编码和相对位置编码结合使用。

---

## 六、位置编码与 KV Cache 的关系

在 decode 阶段，每生成一个新 token，其位置编码对应的是当前序列长度：

```
初始 prompt：[tok_0, tok_1, ..., tok_k]  位置 0, 1, ..., k
第 1 个生成：tok_{k+1}  位置 k+1
第 2 个生成：tok_{k+2}  位置 k+2
...
```

对于 RoPE，每个新 token 的 Q/K 需要用其对应位置的旋转角度：
- 这在 decode 阶段每次只处理一个 token（位置为 current_len）
- 只需计算一个位置的 cos/sin 值，代价极小

---

## 七、总结

| 方法 | 优点 | 缺点 | 代表模型 |
|------|------|------|---------|
| Sinusoidal | 无需训练，理论无限长 | 外推性差 | 原始 Transformer |
| Learned | 简单有效 | 无法外推 | BERT, GPT-2 |
| RoPE | 相对位置，性能好 | 外推需额外技术 | Llama, Qwen, DeepSeek |
| ALiBi | 零样本外推 | 部分任务性能略低 | BLOOM, MPT |

RoPE 凭借其优越的任务性能和与 FlashAttention 等技术的良好兼容性，已成为 LLM 位置编码的事实标准。在长上下文场景中，RoPE 配合 NTK-Aware Scaling 或 YaRN 等外推技术，可以支持 128K-1M token 的超长上下文推理（详见第 33 章）。

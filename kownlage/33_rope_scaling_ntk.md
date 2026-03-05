# 33. RoPE Scaling / NTK-Aware 长度外推

---

## 一、问题背景：RoPE 的外推困境

RoPE（旋转位置编码）是目前最主流的位置编码，但它存在长度外推问题：

**训练长度限制**：模型在训练时只见过最大 $L_\text{train}$ 长度的序列（如 4096 tokens），推理时如果遇到更长的序列，RoPE 的旋转频率对应的位置从未被模型见过，导致性能急剧下降。

**表现**：
- 困惑度（Perplexity）急剧上升
- 模型开始生成重复、不相关的内容
- 严重时模型输出崩溃

---

## 二、RoPE 的数学背景

RoPE 对每个维度 $i$ 使用基频：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2-1$$

位置 $m$ 的旋转角度为 $m\theta_i$。

对于维度 $i$，其"波长"（完整旋转一周所需的 token 数）为：

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot 10000^{2i/d}$$

- 高频维度（$i=0$）：$\lambda_0 = 2\pi$，波长约 6.3（极短）
- 低频维度（$i=d/2-1$）：$\lambda_{d/2-1} = 2\pi \cdot 10000$，波长约 62832

**外推问题的根源**：当序列长度超过某些维度的波长时，这些维度的旋转超过了 $2\pi$，回到了"已见过的"角度，但其他低频维度可能从未在训练数据中出现过对应的旋转角度。

---

## 三、Position Interpolation（位置插值）

最简单的外推方法，Chen et al. 2023 提出。

**思路**：将超出训练长度的位置 $m$ 线性压缩到训练范围 $[0, L_\text{train})$ 内：

$$\text{RoPE}_\text{interpolated}(m) = \text{RoPE}\left(\frac{m \cdot L_\text{train}}{L_\text{target}}\right)$$

其中 $L_\text{target}$ 是目标推理长度。

**效果**：只需要少量微调（1000 步）即可将 Llama 7B 从 2048 扩展到 32768 token。

**问题**：高频维度的旋转角度被压缩，可能损失相邻 token 的相对位置分辨率（近邻 token 的位置变得难以区分）。

---

## 四、NTK-Aware Scaling（神经正切核感知缩放）

NTK-Aware Scaling 是 Reddit 用户 "bloc97" 在 2023 年提出的方法，通过修改 RoPE 的基频 $b$（默认 10000）来实现更好的外推。

### 4.1 核心思想

**关键洞察**：当需要将上下文从 $L$ 扩展到 $\alpha L$ 时，不应该压缩所有维度（Position Interpolation），而应该通过修改基频，使高频维度插值、低频维度外推：

$$b' = b \cdot \alpha^{d/(d-2)}$$

其中：
- $b = 10000$（原始 RoPE 基频）
- $\alpha = L_\text{target} / L_\text{train}$（扩展比例）
- $d$：head_dim

### 4.2 实现

```python
def apply_rope_scaling(model, scale_factor, original_max_seq_len=4096):
    """修改模型的 RoPE 基频（NTK-Aware Scaling）"""
    new_base = 10000 * (scale_factor ** (model.head_dim / (model.head_dim - 2)))

    for layer in model.layers:
        layer.self_attn.rotary_emb.base = new_base
        # 重新计算 cos/sin 缓存
        layer.self_attn.rotary_emb._set_cos_sin_cache(
            seq_len=new_max_len,
            device=layer.self_attn.rotary_emb.inv_freq.device,
        )
```

**优点**：不需要任何微调，推理时动态修改即可使用，是"无训练外推"的最佳方法之一。

### 4.3 NTK-by-Parts

进一步改进：对不同频率的维度使用不同的处理方式：
- **高频维度**（$\lambda_i \ll L$）：不做任何修改（它们已经在训练范围内）
- **中频维度**：应用 NTK Scaling
- **低频维度**（$\lambda_i \gg L$）：应用位置插值

---

## 五、YaRN（Yet another RoPE extensioN method）

YaRN（Peng et al. 2023）是目前效果最好的 RoPE 外推方法之一，被 Llama-3、DeepSeek 等主流模型采用。

### 5.1 YaRN 的核心思路

YaRN 综合了 NTK-by-Parts 的分维度处理思路，并增加了一个关键技巧：

**Temperature Scaling（温度缩放）**：

在 Attention 计算时，对 Softmax 的分母加入一个缩放因子：

$$\text{Attention} = \text{Softmax}\left(\frac{QK^T}{t \cdot \sqrt{d_k}}\right) V$$

其中 $t$ 是温度参数：$t = 0.1 \ln(s) + 1$，$s$ 是扩展比例。

**直觉**：当序列更长时，Attention 权重可能过于"分散"（熵增大），温度缩放可以让模型更专注于重要的 token。

### 5.2 YaRN 的分维度缩放

对每个维度 $i$ 定义不同的缩放系数 $r_i$：

$$r_i = \begin{cases}
1 & \text{如果 } \lambda_i < L_\text{train} / \beta_\text{high} \text{（高频，无需缩放）} \\
\frac{L_\text{train}}{s \cdot \lambda_i} & \text{如果 } \lambda_i > L_\text{train} / \beta_\text{low} \text{（低频，位置插值）} \\
\text{NTK Scaling} & \text{中频维度}
\end{cases}$$

### 5.3 效果

YaRN 可以将 Llama-2 7B（训练长度 4096）扩展到 128K token，且：
- 只需要少量微调（约 400 步）
- 在超出训练长度的序列上，困惑度基本不上升
- 在长文档 QA 任务（如 LongBench）上接近专门训练长上下文模型的效果

---

## 六、LLM 的实际上下文扩展实践

### 6.1 Llama-3 的 RoPE 配置

Meta Llama-3 使用了扩大的基频：
```python
# Llama-3 原始训练：max_seq_len = 8192，base = 500000（更大的基频）
# 对比 Llama-2：base = 10000
```

较大的基频意味着低频维度的波长更长，模型天然对更长序列有更好的"感知"。

**Llama-3 扩展到 128K（Meta 官方）**：
- 使用 YaRN + 微调
- 在 128K 长文档数据上 fine-tune 约 800B tokens

### 6.2 DeepSeek V3 的 RoPE 配置

DeepSeek V3 采用了 YaRN，但对 MLA 的 KV 压缩向量和解耦 RoPE 分别处理：
- 解耦 RoPE 向量：保持完整的位置信息
- 压缩 KV：通过线性层重建 K/V

### 6.3 vLLM 的 RoPE Scaling 配置

```python
from vllm import LLM

# 使用 YaRN 将 Llama-3-8B 扩展到 32K
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=32768,
    rope_scaling={
        "type": "yarn",
        "factor": 8.0,               # 扩展比例（32768 / 4096 = 8）
        "original_max_position_embeddings": 4096,
    }
)

# 使用 NTK-Aware Scaling（无微调外推）
llm = LLM(
    model="...",
    max_model_len=16384,
    rope_scaling={
        "type": "dynamic",           # NTK-Aware Dynamic
        "factor": 4.0,
    }
)
```

---

## 七、各方法对比

| 方法 | 是否需要微调 | 外推质量 | 近邻分辨率 | 适用场景 |
|------|-----------|---------|----------|---------|
| 原始 RoPE | 否 | 差（超过训练长度崩溃） | 好 | 训练长度内 |
| Position Interpolation | 少量微调 | 中 | 略差 | 有微调资源 |
| NTK-Aware | 无需微调 | 良好 | 好 | 零样本外推 |
| NTK-by-Parts | 无需/少量 | 较好 | 好 | 中等扩展 |
| YaRN | 少量微调 | 最好 | 好 | 主流长上下文方案 |

---

## 八、总结

RoPE 外推是支持长上下文 LLM 推理的关键技术：

1. **问题根源**：RoPE 的旋转频率在训练长度之外没有学习到对应的模式
2. **核心思路**：通过修改 RoPE 基频或对位置做缩放，让模型"复用"已学到的位置感知能力
3. **最佳实践**：
   - 无微调场景：NTK-Aware Scaling（即插即用）
   - 有微调条件：YaRN + 少量微调（效果最好）
4. **工程趋势**：越来越多的模型在预训练时就使用更大的基频（如 500K、1M），从根源上支持更长上下文

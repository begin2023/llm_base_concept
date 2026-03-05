# 19. 投机解码（Speculative Decoding）与 MTP

---

## 一、为什么需要投机解码

自回归生成的根本瓶颈：LLM 每次只能生成一个 token，每个 decode step 都需要完整的前向传播。即使 GPU 计算速度很快，由于 decode 是 memory-bound（内存带宽受限），GPU 的计算单元大量空闲。

**核心矛盾**：验证 N 个 token 是否正确（只需一次前向传播）比串行生成 N 个 token（需要 N 次前向传播）要快得多。

投机解码（Speculative Decoding）利用这一不对称性，用更小的"草稿模型"快速生成多个候选 token，再用"目标模型"一次性验证，从而提升整体吞吐量。

---

## 二、经典投机解码（Speculative Decoding）

### 2.1 基本流程

经典投机解码（Leviathan et al. 2023, Chen et al. 2023）包含：

- **草稿模型（Draft Model）**：一个小模型（如 68M 参数的 LLaMA 草稿模型），速度快
- **目标模型（Target Model）**：真正需要推理的大模型（如 70B 参数）

**流程**：

```
Step 1: 草稿模型串行生成 K 个候选 token（draft tokens）
        x1, x2, ..., xK  ← draft model 自回归生成

Step 2: 目标模型对所有 K+1 个位置并行计算 logits
        [p(x1|ctx), p(x2|ctx,x1), ..., p(xK+1|ctx,x1..xK)]
        ← 一次前向传播，计算 K+1 个位置的概率分布

Step 3: 验证（Rejection Sampling）
        对每个 draft token xi，以概率 min(1, p_target(xi) / p_draft(xi)) 接受
        - 如果接受：继续验证下一个 token
        - 如果拒绝：从修正分布中采样一个新 token，停止验证

Step 4: 返回所有被接受的 token（可能是 1 到 K+1 个）
```

### 2.2 关键性质

**无损性（Lossless）**：投机解码在数学上保证，最终生成的 token 序列的分布与目标模型直接生成的分布完全一致（通过拒绝采样实现）。这意味着：
- 不降低生成质量
- 是纯粹的加速技术

**接受率（Acceptance Rate）**：
- 如果草稿模型与目标模型高度对齐，接受率高（接近 1），加速效果好
- 如果对齐度低，接受率低，加速效果差甚至反而更慢

**理论加速比**：
- 假设每次生成平均接受 α 个 draft token，目标模型执行一次前向传播的代价为 T_target
- 草稿模型生成 K 个 token 的代价为 K × T_draft
- 加速比 ≈ (1 + α) / (1 + α × T_draft/T_target)
- 当 T_draft << T_target 且 α 较高时，加速比接近 1 + α

---

## 三、无草稿模型的投机解码变体

### 3.1 Medusa

Medusa 在原始模型的 LM Head 旁边添加多个额外的 Head（"Medusa Heads"）：

```
原始模型最后一层 hidden state
    ├── LM Head 0（原始）→ 预测 x_{t+1}
    ├── LM Head 1（新增）→ 预测 x_{t+2}
    ├── LM Head 2（新增）→ 预测 x_{t+3}
    └── LM Head K（新增）→ 预测 x_{t+K+1}
```

- 这些 Medusa Head 通过在目标模型的数据上微调学习
- 推理时，一次前向传播同时预测未来 K+1 个 token
- 使用树形验证（Tree-based Speculation）来高效验证多条候选路径

**优点**：不需要额外的草稿模型，只需训练几个小 Head
**缺点**：需要微调，接受率通常低于有专用草稿模型的方案

### 3.2 EAGLE / EAGLE-2

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）：

- 在目标模型的 feature（最后一层 hidden state）基础上，训练一个轻量级自回归模型
- 草稿模型接受模型的 feature 作为额外输入，预测能力更强
- EAGLE-2 进一步引入动态草稿长度，根据 feature 相似度自适应决定 K

**性能**：EAGLE-2 在 Llama-2-7B 上实现了约 3-5× 加速。

### 3.3 Lookahead Decoding

完全不需要草稿模型，通过"向前看"的方式利用 n-gram 并行：

- 使用 Jacobi 迭代并行更新多个 token
- 收集高质量的 n-gram 作为候选
- 适合无法训练草稿模型的场景

---

## 四、MTP（Multi-Token Prediction）投机解码

### 4.1 什么是 MTP

MTP（Multi-Token Prediction）最初是 Meta FAIR 在 2024 年论文《Better & Faster Large Language Models via Multi-Token Prediction》中提出的训练目标，但后来被 DeepSeek 等团队创新性地用于推理加速。

**训练目标**：在标准的 next-token prediction 之外，同时训练模型预测未来第 2、3、...、K 个 token。

```
输入序列: [x1, x2, x3, ..., xn]
Head 1（标准）: 预测 x_{t+1}（next token）
Head 2（MTP）:  预测 x_{t+2}
Head 3（MTP）:  预测 x_{t+3}
...
Head K（MTP）:  预测 x_{t+K}
```

### 4.2 DeepSeek V3 的 MTP 实现

DeepSeek V3 在模型架构中引入了 MTP Module：

```
主模型（Main Model）
    ↓ (每层 transformer block)
最终 hidden state h_t
    ├── 标准 LM Head: 预测 x_{t+1}
    └── MTP Module 1:
           ├── 接收 h_t 和 x_{t+1} 的 embedding
           ├── 经过一个 transformer layer
           └── 预测 x_{t+2}
                └── MTP Module 2:
                       ├── 接收前面的 hidden state 和 x_{t+2} 的 embedding
                       └── 预测 x_{t+3}
```

**关键设计**：
- MTP 模块轻量（只有一个 transformer layer）
- MTP 模块可以访问主模型的 hidden state
- 训练时 MTP loss 作为辅助 loss（帮助主模型学习更好的表示）
- 推理时 MTP 模块作为草稿模型使用

### 4.3 MTP 用于推理加速（投机解码）

在推理时，DeepSeek 的 MTP 模块充当内置的草稿模型：

```
前向传播一次:
1. 主模型计算得到 h_t，预测 x_{t+1}
2. MTP Module 1 预测 x_{t+2}（草稿）
3. MTP Module 2 预测 x_{t+3}（草稿）

→ 得到 K+1 个候选 token（1 个确定 + K 个草稿）

验证:
- 主模型以 x_{t+1}（MTP 预测值）为输入，再做一次前向
- 同时验证 MTP 预测的 x_{t+2}, x_{t+3}
- 接受率高时，每次前向传播有效生成 2-3 个 token
```

**DeepSeek V3 的 MTP 数量**：使用了 1 个 MTP 模块（DeepSeek V3），预测未来 1 步。

### 4.4 MTP3 变体

MTP3 是 MTP 的扩展，使用 3 个 MTP 模块，可以同时预测未来 3 个 token。接受率足够高时，理论上每次前向传播可有效生成 3-4 个 token。

---

## 五、树形投机解码（Tree-based Speculation）

当有多个候选 token 时，可以构建候选树：

```
       root
      /    \
    x1a    x1b      ← draft token 1 的两个候选
   /  \   /  \
 x2a x2b x2c x2d   ← draft token 2 的候选
```

验证时，对树中所有路径一次性计算 attention（使用特殊的 tree attention mask），找到最长被接受的路径。

**SpecInfer、Medusa 等框架**采用树形投机提高接受率。

---

## 六、在 vLLM 和 SGLang 中的实现

### 6.1 vLLM 中的投机解码

vLLM 支持多种投机解码策略：

```python
# 使用草稿模型
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,  # K=5 个草稿 token
)

# 使用 EAGLE
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    speculative_model="[eagle]lm_sys/vicuna-7b-v1.3",
    num_speculative_tokens=5,
)

# 使用 n-gram 草稿（无需草稿模型）
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)
```

### 6.2 SGLang 中的投机解码

SGLang 同样支持投机解码，并与 FlashInfer 的高效 attention 深度结合，对 Medusa 有良好支持。

---

## 七、性能分析

### 7.1 加速条件

投机解码加速效果取决于：

1. **接受率（α）**：取决于草稿模型与目标模型的对齐程度
   - 续写任务（小说、代码补全）：接受率通常较高（0.7-0.9）
   - 开放式生成：接受率较低（0.5-0.7）
   - 领域外任务：接受率可能很低

2. **草稿生成速度**：T_draft / T_target 越小越好

3. **批大小**：
   - 小 batch（1-4）：投机解码加速效果最好
   - 大 batch（64+）：GPU 已经满载，投机解码反而可能降低吞吐量

4. **序列长度**：长序列时 KV Cache 更大，内存带宽瓶颈更突出，投机解码改善更明显

### 7.2 典型加速数据

| 方法 | 场景 | 加速比 |
|------|------|--------|
| 标准投机解码（7B + 68M） | 代码生成 | 2-3× |
| EAGLE-2（7B） | 对话 | 3-5× |
| Medusa（7B） | 对话 | 2-3× |
| DeepSeek MTP（671B） | 通用 | 1.8-2× |

---

## 八、总结

| 方案 | 需要额外模型 | 无损 | 适用场景 |
|------|------------|------|---------|
| 标准投机解码 | 是（草稿模型） | 是 | 有配套草稿模型 |
| Medusa | 需要微调 Head | 近似 | 不想训练完整草稿模型 |
| EAGLE | 需要微调轻量层 | 是 | 追求高接受率 |
| Lookahead | 否 | 是 | 无法训练/部署草稿模型 |
| DeepSeek MTP | 内置于模型 | 是 | DeepSeek 系列模型 |

投机解码的核心价值：通过并行验证打破自回归串行生成的瓶颈，在不改变输出分布的前提下实现 2-5× 的推理加速，是目前最有效的 LLM 推理加速技术之一。

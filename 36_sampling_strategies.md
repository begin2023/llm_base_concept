# 36. Sampling 策略（Temperature、Top-K、Top-P、Min-P）

---

## 一、Sampling 的基本原理

LLM 每次生成 token 时，输出一个 vocab_size 维的 logits 向量，经过 softmax 转换为概率分布，然后从中采样下一个 token。

**Greedy Decoding（贪心解码）**：
```python
next_token = argmax(logits)  # 始终选概率最大的 token
```
- 确定性输出，相同输入永远得到相同输出
- 生成内容往往过于保守和重复
- 适合：分类任务、代码生成（要求准确性）

**Sampling（采样）**：
```python
probs = softmax(logits)
next_token = torch.multinomial(probs, num_samples=1)
```
- 随机性输出，增加多样性
- 高概率 token 仍然更可能被选中，但低概率 token 也有机会
- 适合：创意写作、对话生成

---

## 二、Temperature（温度）

Temperature 是控制输出随机性最基本的参数。

### 2.1 数学定义

将 logits 除以温度 T，然后做 softmax：

$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

### 2.2 温度的效果

**T < 1（低温，"冷"）**：
- 放大 logits 之间的差距
- 高概率 token 的概率进一步提升
- 分布更"尖锐"，输出更保守/可预测
- T → 0 时趋近于贪心解码

**T = 1（标准温度）**：
- 使用原始概率分布
- 模型的"自然"随机性

**T > 1（高温，"热"）**：
- 缩小 logits 之间的差距
- 分布更"平坦"，低概率 token 获得更多机会
- 输出更随机/创意，但可能质量下降
- T → ∞ 时趋近于均匀分布（完全随机）

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1, -0.5])

# T=0.5（低温）
probs_cold = F.softmax(logits / 0.5, dim=-1)
print(probs_cold)  # [0.88, 0.10, 0.01, 0.00]（更集中）

# T=1.0（标准）
probs_std = F.softmax(logits / 1.0, dim=-1)
print(probs_std)   # [0.57, 0.31, 0.09, 0.03]

# T=2.0（高温）
probs_hot = F.softmax(logits / 2.0, dim=-1)
print(probs_hot)   # [0.39, 0.30, 0.20, 0.11]（更均匀）
```

### 2.3 典型 Temperature 值

| 场景 | 推荐 Temperature |
|------|----------------|
| 代码生成 | 0.1 - 0.3 |
| 事实问答 | 0.1 - 0.5 |
| 对话（自然） | 0.6 - 0.8 |
| 创意写作 | 0.8 - 1.2 |
| 头脑风暴（多样性） | 1.0 - 1.5 |

---

## 三、Top-K Sampling

Top-K 采样只从概率最高的 K 个 token 中采样，丢弃其余 token。

### 3.1 工作原理

```python
def top_k_filtering(logits, k):
    # 找到第 K 大的值
    kth_value = torch.topk(logits, k).values[-1]
    # 将小于第 K 大值的 logit 设为 -inf
    filtered = logits.clone()
    filtered[filtered < kth_value] = float('-inf')
    return filtered

# 使用
filtered_logits = top_k_filtering(logits, k=50)
probs = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

### 3.2 Top-K 的问题

**问题**：K 是固定的，但不同 token 位置的"自然"分布宽度不同：
- 有些位置几乎只有 1 个合理 token（K=50 包含了很多垃圾 token）
- 有些位置有 100 个合理 token（K=50 截断了很多好 token）

固定的 K 无法自适应这种变化，导致在某些情况下过于保守，在其他情况下过于随机。

### 3.3 常用 Top-K 值

- `k=50`：GPT-2 默认值，较为宽松
- `k=10-20`：较为保守
- `k=1`：等价于贪心解码

---

## 四、Top-P Sampling（Nucleus Sampling，核采样）

Top-P（Nucleus Sampling，Holtzman et al. 2020）通过动态选择候选集合解决了 Top-K 的问题。

### 4.1 工作原理

选择满足"累积概率 ≤ P"的最小 token 集合（"核"），从该集合中采样：

```python
def top_p_filtering(logits, p):
    # 按概率降序排列
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积概率超过 p 的位置
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p

    # 将超出的 token 的 logit 设为 -inf
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    return filtered_logits
```

### 4.2 Top-P 的自适应特性

```
示例：
位置 A（确定性强）：
  token1: 0.95, token2: 0.04, token3: 0.01, ...
  Top-P=0.9 → 只选 token1（cumsum: 0.95 > 0.9）
  → 候选集大小=1，非常保守

位置 B（多样性强）：
  token1: 0.1, token2: 0.09, token3: 0.08, ..., token15: 0.01
  Top-P=0.9 → 选择前 15 个 token（累积到 90%）
  → 候选集大小=15，保留多样性
```

Top-P 自适应地根据分布形状调整候选集大小，这是其优于 Top-K 的核心优势。

### 4.3 典型 Top-P 值

- `p=0.95`：推荐默认值（Llama、ChatGPT 等常用）
- `p=0.90`：稍微保守
- `p=0.99`：非常宽松，接近无截断
- `p=1.0`：不截断

---

## 五、Min-P Sampling

Min-P 是更近期（2024 年）提出的采样方法，解决了 Top-P 在极端情况下的问题。

### 5.1 Top-P 的问题

Top-P 在分布极度集中时可能截断过多：
```
极端情况：
  top_token: 0.999, other tokens: 0.001/vocab_size

  Top-P=0.95：只选 top_token（概率 0.999 > 0.95）
  → 实际上退化为贪心解码，毫无随机性
```

Top-P 在高概率 token 存在时会过于贪心，失去多样性。

### 5.2 Min-P 的工作原理

Min-P 不设置累积概率上限，而是设置**相对于最高概率 token 的最低概率阈值**：

```
min_p = 0.1（设置最低概率阈值为最高概率的 10%）

如果 top_token 概率 = 0.5：
  阈值 = 0.5 × 0.1 = 0.05
  保留所有概率 ≥ 0.05 的 token

如果 top_token 概率 = 0.9：
  阈值 = 0.9 × 0.1 = 0.09
  保留所有概率 ≥ 0.09 的 token（通常只有 1-3 个）

如果 top_token 概率 = 0.1（分布均匀）：
  阈值 = 0.1 × 0.1 = 0.01
  保留很多 token（分布均匀时更宽松）
```

```python
def min_p_filtering(logits, min_p):
    probs = F.softmax(logits, dim=-1)
    # 最高概率值
    top_prob = probs.max()
    # 阈值 = min_p × top_prob
    threshold = min_p * top_prob
    # 过滤掉低于阈值的 token
    filtered_logits = logits.clone()
    filtered_logits[probs < threshold] = float('-inf')
    return filtered_logits
```

### 5.3 Min-P 的优势

- 当分布集中时（top_token 概率高）：阈值高，候选集小，不会乱选低质量 token
- 当分布均匀时（所有 token 概率相近）：阈值低，候选集大，保留多样性
- 比 Top-P 在高温度采样下表现更好

---

## 六、其他 Sampling 技术

### 6.1 Typical Sampling

保留"典型"token（与当前上下文信息熵接近的 token），丢弃极高概率和极低概率的 token。

### 6.2 Epsilon Sampling

设置概率的绝对下限（不是相对于最大值的比例）：
- `epsilon = 0.001`：丢弃概率 < 0.1% 的 token

### 6.3 Repetition Penalty（重复惩罚）

对已生成的 token 施加惩罚，降低重复生成相同 token 的概率：

```python
def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    for token_id in set(input_ids):
        if logits[token_id] < 0:
            logits[token_id] *= penalty      # 负值更负（惩罚更大）
        else:
            logits[token_id] /= penalty      # 正值减小
    return logits
```

### 6.4 Frequency / Presence Penalty（OpenAI 参数）

- **Frequency Penalty**：按 token 出现频率惩罚（出现越多惩罚越大）
- **Presence Penalty**：只要 token 出现过就惩罚（与频率无关）

---

## 七、组合使用

实践中通常组合多种策略：

```python
from vllm import SamplingParams

# 典型的聊天场景配置
params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    # top_k=50,  # 可选
    # min_p=0.05,  # 可选（新版 vLLM 支持）
    repetition_penalty=1.1,
    max_tokens=2048,
)

# 代码生成配置
code_params = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=1024,
)

# 创意写作配置
creative_params = SamplingParams(
    temperature=1.1,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15,
    max_tokens=4096,
)
```

**常见组合策略**：
- Temperature + Top-P（最常见，互相配合）
- Temperature + Top-K（简单有效）
- Temperature + Min-P（新兴方案，效果较好）

---

## 八、Sampling 策略对推理系统的影响

### 8.1 确定性 vs 随机性

- 相同请求重复发送：Sampling 会产生不同结果（seed 固定可复现）
- `temperature=0` 或 `do_sample=False`：退化为贪心解码，确定性输出

### 8.2 Batch 内的独立采样

在 Continuous Batching 中，同一 batch 内的不同请求可以有不同的 sampling 参数，每个请求独立采样。

### 8.3 Speculative Decoding 中的 Sampling

投机解码中，草稿模型的采样策略需要与目标模型对齐，使用拒绝采样（Rejection Sampling）保证分布等价性（见第 19 章）。

---

## 九、总结

| 参数 | 作用 | 推荐范围 |
|------|------|---------|
| Temperature | 整体随机性控制 | 0.1 - 1.5 |
| Top-K | 截断尾部 token（固定数量） | 10 - 100 |
| Top-P | 截断尾部 token（动态数量）| 0.9 - 0.99 |
| Min-P | 相对阈值截断 | 0.01 - 0.1 |
| Repetition Penalty | 减少重复 | 1.0 - 1.3 |

最佳实践：
1. 默认推荐：`temperature=0.7, top_p=0.95`
2. 追求多样性：提高 temperature（0.8-1.2）
3. 追求准确性：降低 temperature（0.1-0.3）
4. 减少重复：添加 `repetition_penalty=1.1-1.2`

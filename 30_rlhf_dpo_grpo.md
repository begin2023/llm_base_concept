# 30. RLHF / DPO / GRPO

---

## 一、背景：为什么需要对齐训练

预训练的 LLM 只是学习了"预测下一个 token"的能力，并不能保证它的输出是有益的、无害的、真实的（HHH：Helpful, Harmless, Honest）。

**对齐问题（Alignment Problem）**：
- 模型可能生成有害内容
- 模型可能说谎（幻觉）
- 模型可能不遵循指令
- 模型的输出可能不符合人类偏好

**解决方案**：对齐训练（Alignment Training），将人类偏好注入模型。

---

## 二、RLHF（Reinforcement Learning from Human Feedback）

RLHF 是 ChatGPT、InstructGPT 等模型使用的对齐训练范式。

### 2.1 RLHF 的三阶段流程

**阶段一：SFT（Supervised Fine-Tuning，监督微调）**

在高质量的指令-回复数据对上做监督学习：
```
数据：{(instruction_1, response_1), (instruction_2, response_2), ...}
训练目标：最大化 P(response | instruction)（Cross-Entropy Loss）
```

**阶段二：训练 Reward Model（奖励模型，RM）**

1. 对同一指令，让 SFT 模型生成多个不同的回复
2. 人工标注员对这些回复进行偏好排序（哪个更好？）
3. 在偏好数据上训练 Reward Model

Reward Model 训练：
```
数据：{(prompt, response_chosen, response_rejected)} - 偏好对
RM 模型结构：LLM backbone + Scalar Head（输出一个分数）
训练目标（Bradley-Terry 模型）：
  Loss = -log σ(r(x, y_w) - r(x, y_l))
  其中 y_w 是偏好回复，y_l 是不偏好回复
  σ 是 sigmoid 函数
```

**阶段三：PPO 强化学习**

使用 PPO（Proximal Policy Optimization）算法，以 RM 的分数作为奖励，优化 SFT 模型：

```
PPO 目标：
  max_π E[r(x, y)] - β * KL(π || π_sft)

其中：
  π：当前策略（正在训练的模型）
  π_sft：SFT 模型（参考策略）
  r(x, y)：Reward Model 给出的分数
  KL(π || π_sft)：KL 散度，防止模型偏离 SFT 太远
  β：KL 惩罚系数
```

### 2.2 RLHF 的主要问题

1. **系统复杂**：需要同时维护 4 个模型（Actor、Critic、Reward Model、Reference Model）
2. **训练不稳定**：PPO 本身容易崩溃，超参调整困难
3. **奖励攻击（Reward Hacking）**：模型可能学会"欺骗" Reward Model 而不是真正改善
4. **计算开销大**：PPO 需要频繁采样，代价高

---

## 三、DPO（Direct Preference Optimization）

DPO 是 Rafael et al. 2023 提出的 RLHF 替代方案，避免了显式训练 Reward Model 和 PPO 训练。

### 3.1 DPO 的核心思想

**关键洞察**：在 RLHF 的最优解中，可以用模型自身的概率比值来隐式表达奖励：

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

把这个关系代入 Bradley-Terry 偏好模型，可以得到直接优化策略模型的损失函数：

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

**直观理解**：
- 增大被偏好回复（$y_w$）相对于参考模型的对数概率
- 减小不被偏好回复（$y_l$）相对于参考模型的对数概率
- $\beta$ 控制对参考模型的偏离程度

### 3.2 DPO 的训练流程

```python
# DPO 损失计算示例（简化）
def dpo_loss(model, ref_model, batch, beta=0.1):
    prompt = batch["prompt"]
    chosen = batch["chosen"]      # 偏好回复
    rejected = batch["rejected"]  # 不偏好回复

    # 当前模型的对数概率
    policy_chosen_logps = get_log_probs(model, prompt, chosen)
    policy_rejected_logps = get_log_probs(model, prompt, rejected)

    # 参考模型的对数概率（不更新梯度）
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, prompt, chosen)
        ref_rejected_logps = get_log_probs(ref_model, prompt, rejected)

    # 计算 log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # DPO 损失
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss
```

### 3.3 DPO 的优势与局限

**优势**：
- 无需训练 Reward Model（节省计算和标注成本）
- 无需 PPO（训练稳定，超参少）
- 实现简单，与 SFT 类似
- 已被 Llama-3、Qwen 等主流模型采用

**局限**：
- 偏好数据是静态的，不能像 RL 一样动态采样新数据
- 当 $\pi_\theta$ 偏离 $\pi_\text{ref}$ 过多时，数据分布可能失配
- 理论上不如 RLHF 强大（无在线学习）

### 3.4 DPO 变体

- **IPO（Identity Preference Optimization）**：修改损失函数防止过度优化
- **SimPO（Simple Preference Optimization）**：去掉参考模型，用平均对数概率作归一化
- **KTO（Kahneman-Tversky Optimization）**：处理单个回复（非偏好对）的偏好数据

---

## 四、GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek 团队提出的 RL 训练算法，被用于训练 DeepSeek-R1，成为推理模型训练的核心技术。

### 4.1 GRPO 的背景

传统 PPO 在 LLM 训练中需要维护一个 Critic 模型（Value Function），这带来：
- 额外的显存占用（Critic 模型与 Actor 同等大小）
- Critic 训练困难（数值不稳定）

GRPO 的目标：去掉 Critic 模型，同时保留 RL 的在线学习优势。

### 4.2 GRPO 的核心算法

**基本思路**：使用**组内相对奖励**代替绝对奖励（Baseline）

对同一个 prompt，生成 G 个不同的回复，计算它们的奖励：
```
同一 prompt x，生成 G 个回复：{y_1, y_2, ..., y_G}
奖励：{r_1, r_2, ..., r_G}

组内归一化（相对奖励）：
Â_i = (r_i - mean(r)) / std(r)
```

**GRPO 目标函数**：

$$\mathcal{L}_\text{GRPO} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right)\right] - \beta \cdot \text{KL}$$

与 PPO 的区别：
- 用组内均值作为基线（baseline），而非 Critic 网络估计的 Value
- 不需要 Critic 模型
- 保留了 PPO 的 clipping 机制防止策略更新过大

### 4.3 奖励函数设计

GRPO 的奖励函数（Reward Function）设计至关重要。DeepSeek-R1 使用**基于规则的奖励**（Rule-based Reward）：

**对于数学/编程任务**：
```
奖励函数：
r(y) = 1  如果答案正确（与 ground truth 一致）
r(y) = 0  如果答案错误
```

**格式奖励（Format Reward）**：
```
r_format(y) = 0.1  如果输出包含 <think>...</think><answer>...</answer> 格式
r_format(y) = 0    否则
```

**不使用 LLM-as-Judge**（避免奖励攻击）：
- DeepSeek-R1 的一个关键设计是不使用另一个 LLM 来评判答案好坏
- 而是使用可验证的规则（数学答案可以精确判断对错）

### 4.4 GRPO 的"顿悟时刻"（Aha Moment）

DeepSeek-R1 在训练过程中，模型自发地学会了"自我反思"（Self-Reflection）：

```
早期训练：
  模型生成：A = 5, B = 3, 所以 A + B = 8。
  （没有反思，直接给答案）

经过 GRPO 训练后：
  模型生成：
  <think>
  让我重新检查一下...A = 5, B = 3
  所以 A + B = 5 + 3 = 8
  等等，让我再验证一次...对，答案是 8
  </think>
  答案是 8
```

这种自我反思行为没有被显式教授，而是在 RL 训练中自然涌现的。

---

## 五、三种方法对比

| 方法 | 是否需要 RM | 是否在线采样 | 训练稳定性 | 效果 | 适用场景 |
|------|-----------|------------|----------|------|---------|
| RLHF（PPO） | 是 | 是 | 较低 | 最强（理论） | 复杂对齐任务 |
| DPO | 否 | 否（静态数据） | 高 | 良好 | 通用指令对齐 |
| GRPO | 否 | 是 | 中等 | 强（推理任务） | 可验证推理任务 |

---

## 六、实践中的训练框架

### 6.1 DPO 训练（HuggingFace TRL）

```python
from trl import DPOTrainer, DPOConfig

# 准备偏好数据集
dataset = load_dataset("path/to/preference_data")
# 格式：{"prompt": "...", "chosen": "...", "rejected": "..."}

# DPO 训练
training_args = DPOConfig(
    beta=0.1,                        # KL 惩罚系数
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,             # 参考模型（SFT 模型）
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 6.2 GRPO 训练（TRL）

```python
from trl import GRPOTrainer, GRPOConfig

def reward_function(completions, ground_truth):
    """自定义奖励函数"""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == gt else 0.0)
    return rewards

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(
        num_generations=8,           # 每个 prompt 生成 G=8 个回复
        max_new_tokens=512,
        beta=0.001,                  # KL 惩罚
    ),
    reward_funcs=reward_function,
    train_dataset=math_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 七、总结

| 特性 | RLHF | DPO | GRPO |
|------|------|-----|------|
| 发表 | InstructGPT 2022 | ICLR 2024 | DeepSeek-R1 2025 |
| 核心技术 | PPO + RM | 直接偏好优化 | 组相对策略优化 |
| 用于推理模型 | 否 | 否 | 是（DeepSeek-R1） |
| 当前重要性 | 中（被 DPO/GRPO 替代） | 高 | 极高（推理模型训练标准） |

这三种方法代表了 LLM 对齐训练的演进路径：从复杂的 RLHF 到简洁的 DPO，再到专为推理优化的 GRPO，每一步都在简化工程复杂度的同时保留或提升了效果。

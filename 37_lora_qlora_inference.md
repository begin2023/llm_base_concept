# 37. LoRA / QLoRA 推理适配

---

## 一、LoRA 基础

### 1.1 什么是 LoRA

LoRA（Low-Rank Adaptation of Large Language Models，Hu et al. 2021）是最流行的参数高效微调（PEFT）方法。

**核心思想**：冻结预训练模型的权重，通过向模型中注入低秩矩阵来学习任务特定的增量。

**数学原理**：

对于原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 添加一个低秩分解：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$（LoRA 的秩，通常 r=4, 8, 16, 64）
- 训练时：$W_0$ 冻结，只训练 $A$ 和 $B$
- 初始化：$A$ 用随机高斯初始化，$B$ 用零初始化（确保训练初始 $\Delta W = 0$）

**参数节省**：
```
原始参数：d × k
LoRA 参数：d × r + r × k = r × (d + k)

例如 d=4096, k=4096, r=8:
原始：4096 × 4096 = 16.7M
LoRA：8 × (4096 + 4096) = 65K（减少 256×）
```

### 1.2 LoRA 的训练

```python
from peft import get_peft_model, LoraConfig

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子（实际 scale = alpha/r）
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 应用 LoRA 的层
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 包装模型
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# → trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

---

## 二、QLoRA：量化 + LoRA

QLoRA（Dettmers et al. 2023）将模型量化（4-bit）与 LoRA 结合，使得在单卡消费级 GPU（如 24GB RTX 3090）上微调 65B 参数模型成为可能。

### 2.1 QLoRA 的核心技术

**1. NF4（NormalFloat 4-bit）量化**：
- 专为正态分布权重设计的 4-bit 量化
- 比普通 INT4 量化精度更高
- 每个权重只占 4 bit（相比 FP16 节省 75% 显存）

**2. Double Quantization（二次量化）**：
- 对量化常数本身再次量化
- 进一步节省约 0.37 bit/参数

**3. Paged Optimizers（分页优化器）**：
- 使用 NVIDIA 统一内存，在 GPU OOM 时自动将优化器状态移到 CPU 内存
- 避免训练过程中的 OOM 崩溃

```python
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 准备用于 k-bit 训练（处理 LayerNorm 等）
model = prepare_model_for_kbit_training(model)

# 添加 LoRA
lora_config = LoraConfig(r=64, lora_alpha=16, ...)
model = get_peft_model(model, lora_config)
```

---

## 三、LoRA 推理

### 3.1 推理时的两种方式

**方式 1：LoRA Weights Merged（合并权重）**

将 LoRA 权重合并到基础模型权重中：

$$W_\text{merged} = W_0 + BA$$

```python
# 合并 LoRA 权重
merged_model = peft_model.merge_and_unload()
# 现在 merged_model 是标准的 HuggingFace 模型，可以直接推理
merged_model.save_pretrained("merged_model")

# 加载合并后的模型（速度与基础模型相同，无 LoRA 开销）
model = AutoModelForCausalLM.from_pretrained("merged_model")
```

**优点**：推理速度与基础模型完全相同，无额外开销
**缺点**：合并后无法切换到其他 LoRA adapter

**方式 2：LoRA Weights Separate（分离权重）**

保持 LoRA 权重独立，推理时实时计算 $W_0 x + BAx$：

```python
from peft import PeftModel

# 加载带有 LoRA 的模型
model = AutoModelForCausalLM.from_pretrained("base_model")
peft_model = PeftModel.from_pretrained(model, "lora_adapter_path")

# 推理
output = peft_model.generate(input_ids, ...)
```

**优点**：可以热切换不同的 LoRA adapter
**缺点**：推理时有额外的矩阵乘法开销

### 3.2 Multi-LoRA 推理

在生产环境中，可能需要同时为多个微调任务提供服务，每个任务对应一个不同的 LoRA adapter。

**vLLM 的 Multi-LoRA 支持**：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 启用 LoRA 支持
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,          # 最多同时加载 4 个 LoRA adapter
)

# 不同请求使用不同 LoRA
outputs = llm.generate(
    prompts=["医学问题...", "法律问题...", "通用问题..."],
    sampling_params=SamplingParams(max_tokens=256),
    lora_request=[
        LoRARequest("medical_lora", 1, "path/to/medical_lora"),
        LoRARequest("legal_lora", 2, "path/to/legal_lora"),
        None,  # 不使用 LoRA（基础模型）
    ]
)
```

**Multi-LoRA 的技术实现**：
- 同一 batch 中不同请求可以使用不同 LoRA
- LoRA 权重存储在 GPU 显存中（按需加载）
- 在矩阵乘法中，对每个请求应用对应的 LoRA 增量

---

## 四、LoRA 的变体

### 4.1 DoRA（Weight-Decomposed LoRA）

将权重分解为"方向"（方向向量）和"大小"（幅度），分别用 LoRA 更新方向，直接更新大小：

$$W = m \cdot \frac{V + \Delta V}{||V + \Delta V||}$$

DoRA 通常比 LoRA 有更好的微调效果，同等参数量下性能更接近全量微调。

### 4.2 LoRA+

LoRA+ 对 LoRA 的两个矩阵 A 和 B 使用不同的学习率：
- B 的学习率 >> A 的学习率（比例约 16:1）
- 理论上更符合梯度流动的特性

### 4.3 LoRA-FA（Frozen A）

冻结 A 矩阵（随机初始化后不更新），只训练 B 矩阵：
- 进一步减少可训练参数
- 节省激活值显存（不需要存 A 的梯度）

### 4.4 rsLoRA（Rank-Stabilized LoRA）

修改 LoRA 的缩放因子：

$$\Delta W = \frac{\alpha}{\sqrt{r}} BA$$（rsLoRA）vs $$\Delta W = \frac{\alpha}{r} BA$$（原始 LoRA）

rsLoRA 在高秩（r > 16）时训练更稳定。

---

## 五、LoRA 在推理中的显存分析

**基础模型 + LoRA 的显存占用**：

| 组件 | Llama-2-7B（BF16） | LoRA（r=16） |
|------|----------------|------------|
| 基础模型权重 | 14 GB | 14 GB |
| LoRA 权重 | - | ~30 MB |
| KV Cache（2K） | 2 GB | 2 GB |
| **总计** | **16 GB** | **16.03 GB** |

LoRA 权重极小，几乎不增加显存。

**QLoRA 的显存节省**：

| 模型 | FP16 | 4-bit QLoRA | 节省 |
|------|------|-------------|------|
| Llama-2-7B | 14 GB | 4 GB | 71% |
| Llama-2-13B | 26 GB | 7 GB | 73% |
| Llama-2-70B | 140 GB | 35 GB | 75% |

---

## 六、最佳实践

### 6.1 训练时的选择

- **资源充足，追求效果**：全量微调（Full Fine-tuning）
- **单卡 A100/H100 80GB**：LoRA（r=64-128），或 Int8 量化 + LoRA
- **单卡 24GB 消费级**：QLoRA（4-bit + LoRA）
- **单卡 16GB**：QLoRA + 小秩（r=4-16）

### 6.2 推理时的选择

| 场景 | 推荐方案 |
|------|---------|
| 单一任务 | 合并 LoRA，与基础模型相同推理速度 |
| 多任务服务 | Multi-LoRA（vLLM），不合并，热切换 |
| 资源受限 | 量化（INT8/INT4）+ 分离的 LoRA |

### 6.3 目标模块的选择

LoRA 通常应用于 Attention 层（Q/K/V/O）和 FFN 层（gate/up/down）：

```python
# 只 Attention（参数少，效果一般）
target_modules=["q_proj", "v_proj"]

# Attention + FFN（参数更多，效果更好，最常用）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# 全模型（接近全量微调）
target_modules=["all-linear"]
```

---

## 七、总结

| 方法 | 可训练参数比例 | 显存需求 | 推理速度 | 效果 |
|------|-------------|---------|---------|------|
| 全量微调 | 100% | 极高 | 基准 | 最好 |
| LoRA（r=16） | 0.1% | 高 | 基准 | 接近全量 |
| QLoRA（4-bit） | 0.1% | 中等 | 略慢 | 略低于 LoRA |

LoRA 已经成为 LLM 微调的标准方法，vLLM 的 Multi-LoRA 支持使得在生产环境中为多个业务场景提供差异化服务成为可能，同时只需维护一个基础模型 + 多个轻量级 LoRA adapter。

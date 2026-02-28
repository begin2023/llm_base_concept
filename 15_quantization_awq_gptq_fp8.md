# 15. 量化（AWQ / GPTQ / FP8）详解

---

## 一、量化的基本概念

### 1.1 什么是量化

量化（Quantization）是指将模型中的数值（权重、激活值等）从高精度的浮点数（如 FP32、FP16、BF16）转换为低精度的数据类型（如 INT8、INT4、FP8）的过程。

本质上，量化是一种"有损压缩"：用更少的 bit 来表示数值，以换取：
- 更小的模型体积（内存占用减少）
- 更快的推理速度（计算和内存带宽需求降低）
- 更低的硬件成本（可以在更小的 GPU 上运行大模型）

代价是：可能会引入一定程度的精度损失。

### 1.2 为什么量化对 LLM 推理特别重要

以 LLaMA-70B 模型为例：

**FP16 精度下**：
- 参数量：70B（700 亿）
- 每个参数占 2 字节（FP16）
- 模型大小 = 70B × 2B = 140 GB
- 需要至少 2 张 A100-80GB 或 2 张 H100-80GB

**INT4 量化后**：
- 每个参数占 0.5 字节
- 模型大小 = 70B × 0.5B = 35 GB
- 仅需 1 张 A100-80GB 即可运行

LLM 推理的 decode 阶段是 memory-bound（内存带宽受限）的：
- 每个 token 生成需要读取全部模型权重
- 量化后权重体积减小，读取速度加快
- 因此量化可以直接提升 decode 阶段的速度

### 1.3 量化的数学基本原理

最简单的对称均匀量化公式：

```
量化：q = round(x / s)      其中 s 是缩放因子（scale）
反量化：x' = q * s

缩放因子计算：s = max(|x|) / (2^(b-1) - 1)   其中 b 是目标位宽
```

**示例（FP16 → INT8 对称量化）**：
```
原始值：x = [0.5, -1.2, 3.7, -0.8, 2.1]
max(|x|) = 3.7
s = 3.7 / 127 = 0.02913
量化值：q = round([17.16, -41.19, 127.0, -27.46, 72.09])
         = [17, -41, 127, -27, 72]
反量化值：x' = [0.4952, -1.1943, 3.6995, -0.7865, 2.0974]
量化误差：[0.0048, 0.0057, 0.0005, 0.0135, 0.0026]
```

非对称量化还引入了零点（zero point）：
```
量化：q = round(x / s) + z
反量化：x' = (q - z) * s
```

---

## 二、量化分类

### 2.1 按量化时机分类

**（1）训练后量化（Post-Training Quantization, PTQ）**
- 在模型训练完成后进行量化
- 不需要重新训练模型
- 通常需要一小部分校准数据（calibration data）来确定量化参数
- 代表方法：GPTQ、AWQ、SmoothQuant
- 优点：快速、方便；缺点：高压缩比时可能有较大精度损失

**（2）量化感知训练（Quantization-Aware Training, QAT）**
- 在训练过程中模拟量化效果
- 前向传播时使用量化后的权重，反向传播时使用全精度梯度
- 通过 Straight-Through Estimator（STE）处理量化操作的不可导问题
- 代表方法：LLM-QAT
- 优点：精度损失更小；缺点：需要训练过程，成本高

### 2.2 按量化对象分类

**（1）权重量化（Weight Quantization）**
- 只量化模型的权重参数，激活值仍使用 FP16/BF16 计算
- 代表配置：W4A16（权重 INT4，激活 FP16）、W8A16

**（2）激活量化（Activation Quantization）**
- 量化中间的激活值（即各层的输入输出）
- 挑战：激活值是动态的，且可能有异常值（outliers）
- 代表方法：SmoothQuant

**（3）权重 + 激活联合量化**
- 同时量化权重和激活值
- 代表配置：W8A8（权重 INT8，激活 INT8），可利用 GPU INT8 Tensor Core

**（4）KV Cache 量化**
- 量化 Key 和 Value 张量，从 FP16 降低到 INT8 或 FP8
- 主要目的：减少 KV Cache 内存占用，支持更大 batch size 或更长上下文
- vLLM 支持 FP8 KV Cache 量化

---

## 三、GPTQ —— 基于二阶信息的逐层量化

### 3.1 背景：从 OBQ 到 GPTQ

GPTQ 全称 "Accurate Post-Training Quantization for Generative Pre-trained Transformers"，发表于 2023 年 ICLR。

核心思路演进：
- **OBS (1993)**：利用 Hessian 矩阵信息决定删除哪些权重（剪枝）
- **OBQ (2022)**：将 OBS 思路应用到量化，逐行量化权重矩阵
- **GPTQ (2023)**：扩展到超大模型（如 GPT-175B），通过算法优化使其可在几小时内量化数百亿参数

### 3.2 算法原理详解

**目标**：量化线性层权重矩阵 W（shape: [d_out, d_in]），使量化后输出与原始输出误差最小。

形式化：
$$\min \|WX - W_q X\|^2_F$$

等价于：
$$\min (W - W_q) H (W - W_q)^T$$

其中 $H = XX^T$ 是 Hessian 矩阵近似。

**GPTQ 核心步骤**：

```
Step 1: 计算 Hessian 矩阵
  H = 2 × X × X^T（使用 128 个校准样本）
  计算 H^{-1}（Cholesky 分解）

Step 2: 逐列量化（相比 OBQ 的关键改进）
  for col = 0 to d_in - 1:
      W_q[:, col] = quantize(W[:, col])
      error = (W[:, col] - W_q[:, col]) / H^{-1}[col, col]
      W[:, col+1:] -= error * H^{-1}[col, col+1:] / H^{-1}[col, col]

Step 3: 分组处理（block_size=128）
  在每个 block 内部逐列量化和误差补偿
  block 处理完后一次性更新剩余列
```

### 3.3 量化粒度：Group Quantization

GPTQ 通常使用分组量化（group_size=128）：
- 每个 group 有独立的 scale 和 zero_point
- 精度高于 per-tensor，存储少于 per-element

### 3.4 GPTQ 的优缺点

| 维度 | 说明 |
|------|------|
| 优点 | 精度损失小、适用广泛、工具链成熟（AutoGPTQ） |
| 缺点 | 需要校准数据、量化过程较慢（70B 模型需数小时）、实现复杂 |

---

## 四、AWQ —— 激活感知的权重量化

### 4.1 核心思想

AWQ（Activation-Aware Weight Quantization）的核心观察：

> "不是所有权重同等重要！只有约 1% 的'显著权重'（salient weights）对模型输出有决定性影响，而这些显著权重可以通过观察激活值来识别。"

**关键洞察**：
- 若权重 w = 1.0，激活值 x = 100，输出 = 100
- 若 w 被量化为 0.8，输出变成 80，误差 = 20
- 若激活值 x = 0.01，量化为 0.8 后误差仅 0.002
- **结论**：对应大激活值的权重通道更需要被保护

### 4.2 实现方法：缩放因子保护

AWQ 通过引入缩放因子 $s$（per-channel）来保护显著通道，而非混合精度：

$$y = Wx = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot x) = W'x'$$

- 数学上输出不变
- 对显著通道（大激活值）：$s_i > 1$，权重被放大，量化相对误差变小
- 对非显著通道：$s_i \approx 1$，维持原有精度

### 4.3 缩放因子的确定

$$s^* = \arg\min_s \| Q(W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X) - WX \|$$

AWQ 使用简化搜索：$s_i = \text{mean}(|x_i|)^\alpha$，$\alpha \in [0, 1]$（最优 $\alpha$ 通常在 0.4-0.6）

搜索过程极快：只需在校准数据上运行一次前向传播，整个量化过程仅需几分钟到几十分钟。

### 4.4 AWQ vs GPTQ 对比

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 核心方法 | Hessian 逆矩阵误差补偿 | 激活感知的缩放保护 |
| 量化速度 | 较慢（数小时级） | 快（分钟级） |
| 需要校准数据 | 是（128 条） | 是（少量） |
| 需要反向传播 | 否 | 否 |
| INT4 精度 | 优秀 | 优秀（通常略好） |
| 实现复杂度 | 较高 | 较低 |
| 推理 kernel | GPTQ kernel / Marlin | AWQ kernel / Marlin |
| 生态支持 | AutoGPTQ | AutoAWQ |

---

## 五、FP8 量化

### 5.1 FP8 数据格式

FP8 有两种主要变体：

| 格式 | 位结构 | 动态范围 | 精度 | 适用场景 |
|------|--------|---------|------|---------|
| E4M3 | 1+4+3 bits | ±448 | 较高 | 前向传播（权重和激活） |
| E5M2 | 1+5+2 bits | ±57344 | 较低 | 反向传播（梯度） |

**各格式对比**：

| 格式 | 位宽 | 动态范围 | 精度 |
|------|------|---------|------|
| FP32 | 32 | ±3.4e38 | 极高 |
| FP16 | 16 | ±65504 | 高 |
| BF16 | 16 | ±3.4e38 | 中 |
| FP8 E5M2 | 8 | ±57344 | 较低 |
| FP8 E4M3 | 8 | ±448 | 低 |
| INT8 | 8 | -128~127 | 均匀 |
| INT4 | 4 | -8~7 | 很低 |

### 5.2 FP8 vs INT8

**FP8 的优势**：
- 浮点格式天然适应非均匀分布的权重和激活值
- 不需要复杂量化校准
- 更好处理异常值（outliers）
- H100 的 FP8 Tensor Core 性能是 FP16 的 2 倍

### 5.3 Hopper 架构（H100）原生支持

NVIDIA H100 是第一个原生支持 FP8 计算的 GPU：
- **FP8 Tensor Core**：FP8 理论峰值 3958 TFLOPS，FP16 为 1979 TFLOPS，FP8 是 FP16 的 2 倍
- **Transformer Engine**：自动在 FP8 和 FP16 之间切换，管理缩放因子

### 5.4 动态量化 vs 静态量化

**动态量化**：运行时动态计算 scale factor，精度高但有额外开销：
```python
x_max = x.abs().max()
scale = x_max / fp8_max  # fp8_max = 448 for E4M3
x_fp8 = (x / scale).to(fp8)
y = matmul(x_fp8, w_fp8) * (scale_x * scale_w)
```

**静态量化**：离线校准确定 scale，推理时直接使用，无额外运行时开销。

**延迟缩放（Delayed Scaling）**：使用前一次迭代的统计量作为当前缩放因子，NVIDIA Transformer Engine 采用此策略。

### 5.5 量化粒度

| 粒度 | Scale 数量 | 精度 | 存储开销 |
|------|-----------|------|---------|
| Per-Tensor | 1 | 最低 | 最小 |
| Per-Channel | d_out 或 d_in | 中等 | 小 |
| Per-Token | batch_size × seq_len | 中高 | 中 |
| Per-Group | n_elements / group_size | 最高 | 较大 |

### 5.6 FP8 在推理框架中的支持

| 框架 | 支持情况 |
|------|---------|
| vLLM | FP8 权重量化（W8A8）、FP8 KV Cache、静态/动态量化 |
| SGLang | 同样支持，使用 FlashInfer 的 FP8 attention kernel |
| TensorRT-LLM | NVIDIA 原生支持，与 Transformer Engine 深度集成 |

---

## 六、常见量化配置

### 6.1 配置命名规则

`W{x}A{y}` 表示"权重 x-bit，激活 y-bit"

| 配置 | 说明 | 特点 |
|------|------|------|
| W4A16 | 权重 INT4，激活 FP16 | 4 倍压缩，最流行配置，需 Marlin kernel |
| W8A8 INT8 | 权重 INT8，激活 INT8 | 2 倍压缩，可用 INT8 Tensor Core |
| W8A8 FP8 | 权重 FP8，激活 FP8 | H100 最优方案，几乎无损 |
| W4A4 | 权重 INT4，激活 INT4 | 极致压缩，精度损失大，仍在研究 |

### 6.2 性能和精度对比

以 LLaMA-2-70B 在 A100-80GB 上为例：

| 配置 | 模型大小 | GPU 数量 | 吞吐量 | PPL 变化 |
|------|---------|---------|--------|---------|
| FP16 | 140 GB | 2 | 基准（1x） | 基准 |
| FP8 | 70 GB | 1 | ~1.8x | +0.01 |
| W8A8 INT8 | 70 GB | 1 | ~1.6x | +0.05 |
| W4A16（Marlin） | 35 GB | 1 | ~2.0x | +0.1~0.3 |

---

## 七、Marlin Kernel —— 高效的 W4A16 GEMM Kernel

### 7.1 背景：W4A16 的性能挑战

W4A16 的推理过程：读取 INT4 权重 → 反量化为 FP16 → 与 FP16 激活做矩阵乘法

**挑战**：反量化开销可能抵消内存节省带来的收益，导致速度反而不如 FP16。

### 7.2 Marlin 的核心优化

Marlin（Mixed Auto-Regressive Linear kernel）由 IST Austria 和 Neural Magic 团队开发：

**（1）全局内存到共享内存的高效传输**
- 使用异步内存拷贝（cp.async）
- 双缓冲（double buffering）

**（2）即时反量化（Just-in-Time Dequantization）**
- 在 Tensor Core 计算之前才进行反量化
- 反量化和计算流水线化，重叠执行

**（3）寄存器级优化**
- 精细管理 GPU 寄存器，减少寄存器溢出

**（4）Tile 大小优化**
- 针对 LLM decode 阶段的小 batch size 进行特殊优化

### 7.3 Marlin 性能表现

在 A100 GPU 上：
- decode 阶段（小 batch）：Marlin W4A16 比 FP16 快约 **3.5-4 倍**
- 模型大小只有 FP16 的 1/4，推理速度反而更快

### 7.4 Marlin 变体与框架集成

- **GPTQ-Marlin**：专用于 GPTQ 量化
- **AWQ-Marlin**：专用于 AWQ 量化
- vLLM 原生支持：`--quantization gptq_marlin` 或 `--quantization awq_marlin`

---

## 八、量化方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| H100/H200 GPU，追求最佳性价比 | FP8（W8A8） | 原生硬件支持，几乎无损，2x 吞吐 |
| A100 GPU，模型无法放入单卡 | AWQ 或 GPTQ + Marlin | 4x 压缩，Marlin kernel 速度快 |
| 追求极致精度 | FP16/BF16（不量化）或 FP8 | 最高精度 |
| 边缘设备 / 端侧部署 | INT4（GGUF 格式）+ llama.cpp | 支持 CPU 推理 |

---

## 九、总结

1. 量化是 LLM 推理优化的核心技术，在模型大小、推理速度、精度之间提供灵活的权衡空间
2. **GPTQ**：通过二阶 Hessian 信息进行最优误差补偿，精度优秀，但量化过程较慢
3. **AWQ**：通过激活感知的缩放保护显著权重，简洁高效，量化速度快
4. **FP8**：新一代 GPU（H100+）上的最佳量化方案，几乎无损且原生硬件支持
5. **Marlin kernel**：解决了 W4A16 的性能问题，使 INT4 量化在速度上也能获益
6. 量化方案选择需综合考虑硬件平台、精度要求、延迟要求和成本约束
7. NVIDIA Blackwell 架构（B100/B200）将支持 FP4，进一步推动量化技术边界

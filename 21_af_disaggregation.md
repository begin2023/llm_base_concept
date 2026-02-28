# 21. AF 分离（Attention-FFN 分离）

---

## 一、背景：Transformer 的两类计算

Transformer 的每一层由两个主要部分组成：

1. **Attention 模块（Self-Attention）**
   - 计算 Q、K、V 矩阵
   - 计算 Attention 权重（QK^T / sqrt(d_k)）
   - 加权求和 V
   - 特点：需要访问 KV Cache，是**内存带宽密集型**（Memory-bound）

2. **FFN 模块（Feed-Forward Network）**
   - 两层线性变换 + 激活函数（如 SwiGLU、GELU）
   - 参数量通常是 Attention 的 2-4 倍（W_up, W_down, W_gate）
   - 特点：大矩阵乘法，是**计算密集型**（Compute-bound）

对于 MoE（Mixture of Experts）模型，FFN 部分被替换为多个 Expert FFN，每个 token 只激活 top-K 个 Expert，这进一步加剧了两类计算的异构性。

---

## 二、AF 分离的概念

AF 分离（Attention-FFN Disaggregation）是将 Attention 计算和 FFN（或 MoE Expert）计算分离到不同硬件上执行的架构思路。

### 2.1 动机

在 MoE 模型中（如 Mixtral、DeepSeek MoE）：

**Attention 层**：
- 参数量小，但每次推理需要读取所有参数
- 需要访问 KV Cache（随序列长度增长）
- 更偏向内存带宽密集

**MoE Expert 层**：
- Expert 总参数量巨大（如 DeepSeek V3 有 671B 参数，但每次只激活 37B）
- Expert 分布在多个设备上（Expert Parallelism）
- 每个 token 路由到不同的 Expert，存在通信开销（All-to-All）
- Expert 的矩阵乘法是计算密集型

### 2.2 AF 分离的基本架构

```
输入 token
    ↓
┌───────────────────────────────────────────────────────┐
│  Attention 设备（A 设备）                               │
│  - 执行 Self-Attention（Q、K、V 计算 + KV Cache）       │
│  - 适合大显存、高带宽的 GPU                             │
└──────────────────────┬────────────────────────────────┘
                       │ hidden state
                       ↓
┌───────────────────────────────────────────────────────┐
│  FFN/Expert 设备（F 设备）                              │
│  - 执行 FFN 或 MoE Expert 计算                         │
│  - 适合高 FLOPS 的 GPU（或 TPU）                        │
└──────────────────────┬────────────────────────────────┘
                       │ 更新后的 hidden state
                       ↓
                  下一层 Attention
```

---

## 三、为什么 MoE 模型更需要 AF 分离

### 3.1 MoE 的 Expert Parallelism 带来的问题

在 Dense 模型中，每层 FFN 参数量固定，可以用标准的 Tensor Parallelism 处理。
在 MoE 模型中，Expert Parallelism（EP）将不同的 Expert 分配到不同的 GPU：

```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

一个 batch 中的 token 需要通过 All-to-All 通信发送到对应的 Expert
```

**All-to-All 通信开销**：在大规模 EP 中，All-to-All 通信开销显著，可能成为瓶颈。

### 3.2 MoE 的负载不均衡问题

不同 Expert 的激活频率可能不同：
- 热门 Expert：被大量 token 选择，成为计算瓶颈
- 冷门 Expert：被很少 token 选择，GPU 资源浪费

AF 分离可以让 Expert 设备独立扩缩容，解决负载不均衡问题。

---

## 四、具体实现方案

### 4.1 基于 EP 的 MoE 分离

DeepSeek V3 / DeepSeek-R1 的部署架构中采用了类似 AF 分离的思路：

```
┌─────────────────────────────────────────────────────┐
│  Attention 节点（少量 GPU）                           │
│  - 所有层的 Attention 计算                            │
│  - 高显存（存储 KV Cache）                            │
│  - Tensor Parallelism across attention GPUs          │
└───────────────────────┬─────────────────────────────┘
                        │ hidden state（通过 NVLink/IB 传输）
                        ↓
┌─────────────────────────────────────────────────────┐
│  Expert 节点（大量 GPU）                              │
│  - 存储所有 Expert 参数                               │
│  - Expert Parallelism：每个 GPU 负责部分 Expert       │
│  - All-to-All：token 路由到对应 Expert                │
└─────────────────────────────────────────────────────┘
```

### 4.2 DeepSeek 的 EPLB（Expert Parallelism Load Balancing）

DeepSeek 提出了 EPLB 技术来解决 Expert 负载不均衡：
- **虚拟 Expert（Redundant Expert）**：将热门 Expert 复制到多个 GPU 上
- 动态调整路由，将 token 分流到不同副本
- 减少热点 GPU 的计算压力

---

## 五、KVCache 在 AF 分离中的挑战

### 5.1 KV Cache 的归属问题

在 AF 分离架构中：
- KV Cache 属于 Attention 计算的一部分
- Attention 节点需要大显存来存储 KV Cache
- 随着请求数量和序列长度增加，KV Cache 显存需求快速增长

### 5.2 解决方案

**分层 KV Cache**：
- 热数据（近期 token 的 KV）存在 GPU 显存
- 温数据存在 CPU 内存
- 冷数据存在 SSD（NVMe）

**KV Cache 压缩**：
- 量化（INT8/INT4）
- KV Cache 剪枝（只保留重要的 KV）

---

## 六、与 PD 分离的关系

AF 分离和 PD 分离是两个不同维度的分离：

| 维度 | PD 分离 | AF 分离 |
|------|--------|--------|
| 分离对象 | 计算阶段（Prefill vs Decode） | 计算类型（Attention vs FFN） |
| 适用模型 | 所有 LLM | MoE 模型更明显 |
| 主要收益 | 消除 Prefill-Decode 干扰 | 适配异构硬件，解决 MoE 负载不均衡 |
| 通信开销 | KV Cache 传输 | Hidden state 传输 + All-to-All |

两者可以**同时使用**：
- 将 Prefill 和 Decode 分离（PD 分离）
- 在 Decode 节点内部，再将 Attention 和 FFN 分离（AF 分离）

---

## 七、实践中的考量

### 7.1 适用场景

AF 分离主要适用于：
1. **超大规模 MoE 模型**（Expert 数量多，参数量大）
2. **异构集群**（不同型号 GPU，分别适合 Attention 和 Expert 计算）
3. **显存受限场景**（KV Cache 和 Expert 参数都需要大显存）

### 7.2 通信开销的挑战

AF 分离的核心挑战是 Attention 节点和 FFN 节点之间的 hidden state 传输：
- 每层都需要在 A 节点和 F 节点之间传输 hidden state
- hidden state 大小 = batch_size × seq_len × hidden_dim × dtype_bytes
- 需要极高带宽（NVLink 或 InfiniBand）

### 7.3 延迟分析

```
一次 Transformer 层的延迟：
= Attention 计算时间（A 节点）
+ A→F 传输时间（hidden state 发送）
+ FFN/Expert 计算时间（F 节点）+ All-to-All 时间
+ F→A 传输时间（hidden state 接收）
```

如果传输时间可以与下一层计算流水线化（pipeline），延迟可以被大幅隐藏。

---

## 八、总结

AF 分离是针对 MoE 等大型模型的高级部署优化策略：

1. **核心动机**：Attention 和 FFN 计算特性差异大，适合不同类型的硬件
2. **关键挑战**：层间 hidden state 传输带来的通信开销
3. **主要收益**：
   - 支持超大规模 MoE 模型的高效部署
   - 解决 Expert 负载不均衡问题
   - 实现更灵活的资源分配
4. **实际应用**：DeepSeek 的大规模 MoE 模型推理是最典型的应用案例
5. **发展方向**：与 PD 分离结合，构建多维度解耦的推理系统

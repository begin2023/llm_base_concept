# 5. MoE（Mixture of Experts，混合专家模型）详解

---

## 5.1 基本原理

### 5.1.1 什么是 MoE

MoE（Mixture of Experts）是一种**条件计算（Conditional Computation）**架构。其核心思想是：**模型拥有大量参数（专家网络），但每次推理只激活其中一小部分，从而在保持大模型容量的同时控制计算量**。

### 5.1.2 核心组件

MoE 层由两个核心组件组成：

```
┌─────────────────────────────────────────────────────┐
│                    MoE Layer                        │
│                                                     │
│  ┌──────────────┐                                   │
│  │   Gate /      │                                   │
│  │   Router      │──→ 路由决策: 哪些专家处理哪些 token │
│  │  (门控网络)    │                                   │
│  └──────┬───────┘                                   │
│         │                                           │
│  ┌──────┼──────────────────────────────┐            │
│  │      ▼                              │            │
│  │ ┌────────┐ ┌────────┐   ┌────────┐ │            │
│  │ │Expert 1│ │Expert 2│...│Expert N│ │            │
│  │ │  (FFN) │ │  (FFN) │   │  (FFN) │ │            │
│  │ └────────┘ └────────┘   └────────┘ │            │
│  │           专家网络                   │            │
│  └─────────────────────────────────────┘            │
│                                                     │
│  输出 = Σ gate_weight_i × Expert_i(x)              │
└─────────────────────────────────────────────────────┘
```

#### （1）门控网络（Gate / Router）

门控网络是一个轻量级网络（通常是一个线性层 + softmax），负责决定每个 token 应该由哪些专家处理：

```python
# 门控网络的基本实现
class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        logits = self.gate(x)                    # [batch, seq, num_experts]
        scores = F.softmax(logits, dim=-1)       # 归一化为概率

        # 选择 top-k 个专家
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        # 重新归一化选中专家的权重
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        return top_k_scores, top_k_indices
```

#### （2）专家网络（Expert Networks）

每个专家通常是一个标准的 FFN（Feed-Forward Network），结构与 Dense Transformer 中的 FFN 相同：

```python
class Expert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)   # Up projection
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)   # Down projection
        self.w3 = nn.Linear(hidden_dim, intermediate_dim)   # Gate projection (SwiGLU)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))    # SwiGLU activation
```

### 5.1.3 MoE 的数学表达

给定输入 token 表示 x，MoE 层的输出为：

```
y = Σ_{i∈TopK(G(x))} g_i(x) · E_i(x)

其中：
  G(x) = Softmax(W_g · x)           — 门控得分
  g_i(x) = G(x)_i / Σ_{j∈TopK} G(x)_j  — 归一化后的 top-k 门控权重
  E_i(x)                             — 第 i 个专家的输出
  TopK(G(x))                         — 门控得分最高的 k 个专家索引
```

### 5.1.4 MoE 在 Transformer 中的位置

在 Transformer 架构中，MoE 通常**替换 FFN 层**（而非 Attention 层）：

```
标准 Transformer Block:
  Input → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual → Output

MoE Transformer Block:
  Input → LayerNorm → Attention → Residual → LayerNorm → MoE(FFN) → Residual → Output
```

注意：不是每层都用 MoE，通常是**间隔使用**（如每 2 层用一次 MoE）。

---

## 5.2 稀疏激活

### 5.2.1 稀疏激活的核心优势

```
Dense Model (7B):
  每个 token 激活 7B 参数
  需要 7B 参数的计算量

MoE Model (47B 总参数, 8 专家, Top-2):
  每个 token 只激活 ~14B 参数 (共享部分 + 2个专家)
  计算量接近 14B 的 Dense Model
  但拥有 47B 的模型容量（知识存储能力）
```

**关键洞察**：MoE 模型用**更多的参数但相似的计算量**获得更好的性能。这是因为：
- 不同的专家可以学习不同类型的知识/模式
- 模型总容量大，但推理成本低
- 本质上是用"内存/参数量"换"计算量"

### 5.2.2 Top-K Routing 策略

**Top-1 Routing**（最常见于早期模型如 Switch Transformer）：
```
每个 token 只发给 1 个专家
优点: 计算量最小
缺点: 容易出现路由不稳定，部分专家被"饿死"
```

**Top-2 Routing**（GShard、ST-MoE 等）：
```
每个 token 发给得分最高的 2 个专家，加权求和
优点: 更稳定，信息混合更充分
缺点: 计算量是 Top-1 的 2 倍
```

**Top-K Routing（K>2）**（DeepSeek-V2 使用 Top-6）：
```
每个 token 发给得分最高的 K 个专家
配合细粒度专家（更多但更小的专家），可以实现更灵活的组合
```

### 5.2.3 稀疏性的代价

稀疏激活带来的挑战：
1. **负载不均衡**：某些专家被过度使用，其他专家闲置
2. **训练不稳定**：路由决策是离散的，梯度难以传播
3. **Token Dropping**：当专家容量不足时，部分 token 可能被丢弃
4. **通信开销**：分布式环境下 token 需要在不同 GPU 间路由

---

## 5.3 DeepSeek-V2/V3 的 MoE 架构特点

### 5.3.1 DeepSeek-V2 的 DeepSeekMoE

DeepSeek-V2（236B 总参数，21B 激活参数）引入了两个重要创新：

#### （1）细粒度专家（Fine-Grained Experts）

传统 MoE 的专家数量少（如 8-16 个大专家），DeepSeekMoE 将专家分得更细：

```
传统 MoE (如 Mixtral):
  8 个专家, 每个专家参数量 = FFN_dim
  Top-2 routing → 激活 2 个专家

DeepSeek-V2 的 DeepSeekMoE:
  160 个路由专家 (每个更小), Top-6 routing → 激活 6 个小专家
  等效计算量 ≈ 传统 Top-2 + 2个大专家
  但组合更灵活: C(160,6) >> C(8,2)
```

**细粒度的优势**：
- 更多的专家组合方式，模型可以更精细地为不同 token 选择不同的知识组合
- 每个小专家可以更加专业化
- Top-K 选择时有更高的灵活性

```
具体配置 (DeepSeek-V2):
  每层: 2 个共享专家 + 160 个路由专家
  每个路由专家的 intermediate_dim = FFN_dim / 多个分割
  Top-6 routing: 每个 token 选择 6 个路由专家
  总激活参数 ≈ 共享专家 + 6 个小路由专家
```

#### （2）共享专家（Shared Experts）

这是 DeepSeekMoE 的另一个核心创新：

```
标准 MoE:
  所有专家都是路由的（条件激活）
  结果 = Σ gate_i × Expert_i(x)

DeepSeekMoE:
  共享专家: 始终激活，处理所有 token
  路由专家: 条件激活，通过门控选择

  结果 = Σ SharedExpert_j(x) + Σ gate_i × RoutedExpert_i(x)
```

**共享专家的作用**：
- 捕获所有 token 都需要的**通用知识**（如语法规则、基本语义）
- 减少路由专家之间的知识冗余（路由专家不需要重复学习通用知识）
- 提高训练稳定性（即使路由不完美，共享专家也能保证基本输出质量）

```python
class DeepSeekMoELayer(nn.Module):
    def __init__(self, hidden_dim, num_shared_experts, num_routed_experts, top_k):
        super().__init__()
        self.shared_experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim)
            for _ in range(num_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim_small)
            for _ in range(num_routed_experts)
        ])
        self.router = Router(hidden_dim, num_routed_experts, top_k)

    def forward(self, x):
        # 共享专家: 所有 token 都经过
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # 路由专家: 选择 top-k
        scores, indices = self.router(x)
        routed_output = sum(
            scores[..., i:i+1] * self.routed_experts[idx](x)
            for i, idx in enumerate(indices)
        )

        return shared_output + routed_output
```

### 5.3.2 DeepSeek-V3 的进一步创新

DeepSeek-V3（671B 总参数，37B 激活参数）在 V2 的基础上引入了更多创新：

#### （1）无辅助损失的负载均衡（Auxiliary-Loss-Free Load Balancing）

**传统方法的问题**：

传统 MoE 使用辅助损失（auxiliary loss）来强制负载均衡：
```
总损失 = 任务损失 + α × 负载均衡辅助损失

辅助损失 = N × Σ_i (f_i × p_i)
  f_i = 第 i 个专家处理的 token 比例
  p_i = 第 i 个专家的平均路由概率
  α = 辅助损失权重（超参数）
```

辅助损失的缺点：
- α 需要仔细调参：太大会损害模型性能，太小则无法有效均衡
- 辅助损失和任务损失存在冲突
- 会降低模型最终性能

**DeepSeek-V3 的解决方案**：

引入了一个**动态偏置项（bias term）**加到路由得分上，根据专家的历史负载动态调整：

```
实际路由得分 = gate_score + bias_i

其中 bias_i 根据专家 i 的负载动态更新：
  如果专家 i 被过度使用 → 降低 bias_i（让它更难被选中）
  如果专家 i 被不足使用 → 提高 bias_i（让它更容易被选中）
```

这种方式不需要辅助损失，直接通过动态调整路由偏置来实现均衡，**不会干扰任务损失的优化**。

#### （2）多 Token 预测（Multi-Token Prediction, MTP）

DeepSeek-V3 除了标准的 next-token prediction，还训练模型同时预测**未来多个 token**：

```
标准训练目标:
  给定 x_1, ..., x_t → 预测 x_{t+1}

MTP 训练目标:
  给定 x_1, ..., x_t → 同时预测 x_{t+1}, x_{t+2}, ..., x_{t+k}
```

MTP 的优势：
- 模型学到更好的内部表示（需要理解更远的依赖关系）
- 推理时可以用于投机解码（Speculative Decoding），显著提升吞吐量
- 不增加推理时的主干计算量（MTP 头可以在推理时作为 Draft Model）

#### （3）具体架构参数

```
DeepSeek-V3 架构配置:
  总层数: 61 层
  隐藏维度: 7168
  注意力头数: 128
  KV 压缩维度: 512 (MLA)

  MoE 配置 (应用于部分层):
    共享专家: 1 个
    路由专家: 256 个
    Top-K: 8 (每个 token 激活 8 个路由专家)
    每个路由专家的 intermediate_dim: 2048

  总参数: 671B
  每 token 激活参数: 37B
  训练数据: 14.8T tokens
  训练成本: 2.788M H800 GPU-hours
```

---

## 5.4 推理中的挑战

### 5.4.1 负载均衡问题

推理时的负载均衡比训练时更具挑战性：

```
训练时:
  - 大批量数据，负载统计上倾向于均衡
  - 可以使用辅助损失/动态偏置进行调整
  - 可以 drop 超出容量的 token

推理时:
  - 请求动态到达，批次大小变化
  - 不能丢弃任何 token
  - 某些类型的查询可能集中使用特定专家
  - 负载瞬时不均衡很常见
```

**推理场景的负载不均衡后果**：
- 热门专家成为瓶颈，请求排队等待
- 冷门专家的 GPU 资源浪费
- 整体延迟受最慢专家限制（木桶效应）

### 5.4.2 专家并行的通信开销

当模型太大无法放在单个 GPU 上时，需要将不同专家分布到不同 GPU：

```
假设 8 个 GPU，256 个专家:
  GPU 0: 专家 0-31
  GPU 1: 专家 32-63
  ...
  GPU 7: 专家 224-255

一个 token 被路由到专家 5 (GPU 0) 和专家 100 (GPU 3):
  1. Token 从当前 GPU 发送到 GPU 0 和 GPU 3  (All-to-All 通信)
  2. GPU 0 和 GPU 3 分别计算专家输出
  3. 结果发送回原 GPU 并合并            (All-to-All 通信)
```

**通信瓶颈**：
- 每个 MoE 层需要 **2 次 All-to-All 通信**（发送 token + 收集结果）
- All-to-All 通信模式是所有集合通信中最复杂的
- 通信量随 GPU 数量增加而增加
- 跨节点通信延迟远高于节点内通信

### 5.4.3 内存管理挑战

```
MoE 模型的内存需求:
  671B 参数 × 2 bytes (FP16) = 1.34 TB（仅模型权重）

即使用 FP8 量化: 671B × 1 byte = 671 GB

需要至少 8-16 个 80GB GPU 才能装下模型
加上 KV Cache、激活值等，实际需要更多
```

### 5.4.4 专家局部性问题

推理时一个重要的优化方向是**专家局部性**：

```
如果连续的请求大多路由到相同的专家:
  → 这些专家的权重已在 GPU 缓存中 → 快速

如果连续的请求路由到不同的专家:
  → 需要频繁加载不同专家的权重 → 慢（尤其是 offloading 场景）
```

在 CPU/GPU offloading 场景（部分专家放在 CPU 内存中）下，专家局部性尤为重要。

---

## 5.5 Expert Parallelism（EP）策略

### 5.5.1 什么是 Expert Parallelism

Expert Parallelism（专家并行）是专门为 MoE 模型设计的并行策略，将**不同专家分配到不同 GPU** 上：

```
┌────────────────────────────────────────────────────┐
│                    Data Flow                        │
│                                                     │
│  Input Tokens ──→ Self-Attention (各 GPU 副本相同)   │
│       │                                             │
│       ▼                                             │
│  Router/Gate ──→ 路由决策                            │
│       │                                             │
│       ▼                                             │
│  ┌─── All-to-All 通信 (分发 token 到对应专家) ───┐   │
│  │                                               │   │
│  │  GPU 0        GPU 1        GPU 2     GPU 3    │   │
│  │  Expert 0-1   Expert 2-3   Expert 4-5 Expert 6-7 │
│  │  ↓            ↓            ↓         ↓        │   │
│  │  计算         计算          计算       计算     │   │
│  │                                               │   │
│  └─── All-to-All 通信 (收集结果回原 GPU) ────────┘   │
│       │                                             │
│       ▼                                             │
│  Weighted Sum + Residual                            │
└────────────────────────────────────────────────────┘
```

### 5.5.2 EP 与其他并行策略的组合

实际部署中，EP 通常与其他并行策略组合使用：

#### EP + TP (Tensor Parallelism)

```
场景: 单个专家太大，一个 GPU 放不下
方案: 在专家内部做张量并行

例如 8 GPU, 4 专家:
  GPU 0,1: Expert 0 (TP=2)
  GPU 2,3: Expert 1 (TP=2)
  GPU 4,5: Expert 2 (TP=2)
  GPU 6,7: Expert 3 (TP=2)
```

#### EP + DP (Data Parallelism)

```
场景: 需要更高吞吐量
方案: 每组 GPU 放完整的专家集，多组并行处理不同数据

例如 16 GPU, 8 专家:
  组1 (GPU 0-7):  Expert 0-7, 处理数据批次 A
  组2 (GPU 8-15): Expert 0-7, 处理数据批次 B
```

#### EP + PP (Pipeline Parallelism)

```
场景: 模型太深，一组 GPU 放不下所有层
方案: 不同层放在不同 GPU 组

例如 32 GPU, 模型 60 层:
  GPU 0-7:   层 0-14 (EP=8 用于 MoE 层)
  GPU 8-15:  层 15-29 (EP=8 用于 MoE 层)
  GPU 16-23: 层 30-44 (EP=8 用于 MoE 层)
  GPU 24-31: 层 45-59 (EP=8 用于 MoE 层)
```

### 5.5.3 DeepSeek-V3 的分布式策略

DeepSeek-V3 使用了精心设计的混合并行策略：

```
DeepSeek-V3 训练配置:
  Pipeline Parallelism (PP) = 16  (16 个流水线阶段)
  Expert Parallelism (EP) = 64   (专家分布在 64 个 GPU 上)
  Data Parallelism (DP) = 若干   (用于增加吞吐量)

  总计约使用 2048 个 H800 GPU

推理部署:
  由于只需要前向传播，可以使用更灵活的策略
  EP 度数可以根据延迟要求调整
```

### 5.5.4 All-to-All 通信优化

All-to-All 是 EP 的核心通信原语，优化策略包括：

#### （1）分层 All-to-All

```
节点内 All-to-All (使用 NVLink, ~900 GB/s):
  先在同一节点的 GPU 之间交换

节点间 All-to-All (使用 InfiniBand/RDMA, ~400 GB/s):
  再在不同节点之间交换

两步分层通信减少了跨节点的数据量
```

#### （2）通信-计算重叠

```
Layer N 的计算和 Layer N+1 的通信可以重叠：

时间线:
  计算:   [Layer N Attn] [Layer N MoE Expert计算] [Layer N+1 Attn] ...
  通信:                  [Layer N+1 All-to-All]                   ...
```

#### （3）专家容量因子（Capacity Factor）

```
容量因子 C 决定每个专家最多处理多少 token:
  expert_capacity = C × (total_tokens / num_experts) × top_k

C = 1.0: 完美均衡时刚好装满
C = 1.25: 允许 25% 的不均衡
C > 1.0: 更多内存开销，但更少 token 被丢弃
```

### 5.5.5 推理时的 EP 优化技巧

```
1. Expert Offloading:
   冷门专家放在 CPU 内存，热门专家常驻 GPU
   根据访问频率动态调整

2. Expert Caching:
   缓存最近使用的专家权重
   利用局部性原理减少数据搬运

3. 专家预取 (Prefetching):
   在 Router 计算完成后立即开始预取下一层需要的专家权重
   与当前层的计算重叠

4. 批次级别路由聚合:
   将多个请求的路由需求聚合
   减少 All-to-All 通信次数
   提高专家利用率
```

---

## 5.6 MoE 模型一览

| 模型 | 总参数 | 激活参数 | 专家数 | Top-K | 共享专家 | 细粒度 |
|------|--------|---------|--------|-------|---------|--------|
| Switch Transformer | - | - | 128 | 1 | 无 | 否 |
| GShard | 600B | - | 2048 | 2 | 无 | 否 |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | 无 | 否 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 | 无 | 否 |
| DeepSeek-V2 | 236B | 21B | 160(+2) | 6 | 2 | 是 |
| DeepSeek-V3 | 671B | 37B | 256(+1) | 8 | 1 | 是 |
| Qwen2-MoE | 57B | 14.3B | 64 | 8 | 8 | 是 |

---

## 5.7 MoE 的优缺点总结

### 优点

1. **参数效率**：用少量计算获得大模型的性能
2. **推理速度**：激活参数少，推理计算量可控
3. **可扩展性**：增加专家数量即可扩展模型容量
4. **专业化**：不同专家可以学习不同领域的知识

### 缺点

1. **内存占用大**：所有专家权重都需要在 GPU 内存中
2. **通信开销**：EP 中的 All-to-All 通信成为瓶颈
3. **负载均衡困难**：热门专家过载，冷门专家浪费
4. **训练复杂性**：路由学习不稳定，需要特殊技巧
5. **批次效率**：不同 token 路由到不同专家，难以高效批处理

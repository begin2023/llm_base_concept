# 20. PD 分离（Prefill-Decode 分离）

---

## 一、背景与动机

### 1.1 Prefill 和 Decode 的根本差异

在 LLM 推理中，一次请求分为两个阶段：

**Prefill 阶段**（预填充）：
- 输入：用户 prompt（可能有几十到几千个 token）
- 输出：第一个生成 token，以及所有 prompt token 的 KV Cache
- 计算特征：**Compute-bound（计算密集型）**
  - 每个 token 都要与所有其他 token 做 attention
  - 矩阵乘法的操作数量 = O(seq_len²)
  - GPU 的 FLOPs 充分利用
  - arithmetic intensity（算术强度，FLOPs/Byte）高

**Decode 阶段**（解码）：
- 输入：之前生成的 token + KV Cache
- 输出：下一个 token
- 计算特征：**Memory-bound（内存带宽密集型）**
  - 每次只处理一个 token，但需要读取所有层的 KV Cache
  - GPU 的计算单元大量空闲，瓶颈在显存带宽
  - arithmetic intensity 低

### 1.2 混合部署的问题

在传统的混合部署（Prefill 和 Decode 在同一 GPU 上）中：

**干扰问题（Interference）**：
- Prefill 是 compute-bound，会占用大量 GPU 计算资源
- 当 Prefill 和 Decode 请求混合在同一批次（batch）中时，Decode 请求需要等待 Prefill 完成
- 导致 Decode 请求的 TPOT（Time Per Output Token）增大，尾延迟恶化

**资源利用不匹配**：
- 专注 Prefill 的 GPU 需要高计算能力（多核心，高 FLOPS）
- 专注 Decode 的 GPU 需要高内存带宽和大显存
- 同一型号 GPU 难以同时优化两种需求

---

## 二、PD 分离架构

### 2.1 基本思想

PD 分离（Prefill-Decode Disaggregation）将 Prefill 和 Decode 阶段分配到不同的 GPU（或机器）上：

```
客户端请求
    ↓
Router / Load Balancer
    ↓
┌─────────────────┐      KV Cache 传输      ┌─────────────────┐
│  Prefill 节点   │  ──────────────────→   │  Decode 节点    │
│  (P 节点)       │                         │  (D 节点)       │
│                 │                         │                 │
│ - 处理 prompt   │                         │ - 自回归生成    │
│ - Compute-bound │                         │ - Memory-bound  │
│ - 高 FLOPS GPU  │                         │ - 大显存/高带宽 │
└─────────────────┘                         └─────────────────┘
```

**工作流程**：
1. 请求到达 Router，被发送到 P 节点
2. P 节点执行 Prefill，生成第一个 token 和 KV Cache
3. KV Cache 通过高速网络（RDMA/NVLink）传输到 D 节点
4. D 节点接收 KV Cache，开始 Decode 阶段
5. 生成的 token 流式返回给客户端

### 2.2 系统架构详解

```
┌──────────────────────────────────────────────────────────┐
│                      PD 分离系统                          │
│                                                          │
│  ┌────────────┐    ┌────────────────────────────────┐    │
│  │   Router   │    │        P 节点集群               │    │
│  │            │→→→│  P1: prefill worker             │    │
│  │ - 请求路由  │    │  P2: prefill worker             │    │
│  │ - 负载均衡  │    │  P3: prefill worker             │    │
│  │            │    └──────────────┬─────────────────┘    │
│  │            │                   │ KV Cache Transfer    │
│  │            │    ┌──────────────▼─────────────────┐    │
│  │            │←←←│        D 节点集群               │    │
│  │            │    │  D1: decode worker              │    │
│  └────────────┘    │  D2: decode worker              │    │
│                    │  D3: decode worker              │    │
│                    └────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 三、KV Cache 传输

PD 分离的核心技术挑战是高效的 KV Cache 传输。

### 3.1 KV Cache 的大小

KV Cache 大小估算：
```
KV Cache size per token = 2 × num_layers × num_heads × head_dim × dtype_bytes
例如 Llama-2-7B (BF16):
= 2 × 32 × 32 × 128 × 2 bytes
= 524,288 bytes ≈ 512 KB per token

对于 1000 token 的 prompt：
KV Cache = 512 MB
```

这意味着 KV Cache 传输需要高带宽，普通以太网（10 GbE）远远不够。

### 3.2 传输技术方案

**RDMA（Remote Direct Memory Access）**：
- 绕过 CPU，直接将 GPU 显存中的 KV Cache 传输到远端节点
- 使用 InfiniBand 或 RoCE 网络
- 带宽可达 200 Gbps（HDR InfiniBand）
- 传输 512 MB KV Cache 的时间约 20ms

**NVLink + NVSwitch**：
- 对于同一机箱内的 GPU（如 NVL32、GB200 NVL72）
- NVLink 带宽达 900 GB/s（H100 NVLink）
- 几乎可以认为是"本地"传输
- 适合 P 节点和 D 节点在同一机箱的场景

**KV Cache 压缩**：
- 在传输前对 KV Cache 进行量化（如 INT8）压缩
- 可将传输数据量减半
- 接收端反量化后使用

### 3.3 传输对延迟的影响

传输延迟加入 TTFT（首 Token 延迟）：
```
TTFT = Prefill 计算时间 + KV Cache 传输时间 + Decode 第一步时间
```

如果 P 节点和 D 节点在同一机箱（NVLink），传输延迟可以忽略。
如果跨机器（RDMA），需要仔细设计来最小化传输延迟。

---

## 四、调度策略

### 4.1 P 节点调度

P 节点专注于处理 Prefill 请求：
- 维护自己的 Continuous Batching
- 可以将多个 prefill 请求批处理，充分利用 compute 能力
- 处理完成后，立即将 KV Cache 推送到 D 节点

### 4.2 D 节点调度

D 节点专注于 Decode：
- 接收来自 P 节点的 KV Cache，开始 decode
- 所有请求都处于 decode 阶段，batch 均匀，内存带宽利用率高
- 通过 Continuous Batching 最大化吞吐量

### 4.3 P/D 节点比例

P 节点和 D 节点的最优比例取决于工作负载：

- **长 prompt，短输出**：需要更多 P 节点（prefill 是瓶颈）
- **短 prompt，长输出**：需要更多 D 节点（decode 是瓶颈）
- **动态调整**：有些系统支持根据实时负载动态调整 P/D 比例（"弹性 PD 分离"）

典型比例：P:D = 1:3 到 1:10（decode 需要更多资源）

---

## 五、异构部署

PD 分离还允许为 P 节点和 D 节点使用不同类型的硬件：

### 5.1 P 节点：计算型 GPU

- **特点**：需要高 FLOPS，显存需求相对较小
- **适合硬件**：H100 SXM（高 FLOPS）、A100
- **优化方向**：最大化矩阵乘法吞吐量

### 5.2 D 节点：内存型 GPU

- **特点**：需要大显存和高内存带宽
- **适合硬件**：
  - H100 NVL（更大显存）
  - 专为推理设计的加速卡
  - 甚至 CPU（对于极低 QPS 的场景）
- **优化方向**：最大化内存带宽

### 5.3 成本优化

- D 节点可以使用成本更低的 GPU（不需要最高 FLOPS）
- 通过异构部署降低整体推理成本

---

## 六、代表性系统实现

### 6.1 Mooncake（月之暗面）

Mooncake 是月之暗面（Kimi）开源的 PD 分离系统，特点：
- 提出了以 KV Cache 为中心的调度架构
- KV Cache 传输使用 RDMA
- 支持 KV Cache 的多级缓存（GPU → CPU → SSD）
- 发表在 2024 年 OSDI

### 6.2 DistServe

DistServe（CMU 2024）：
- 研究了 P/D 分离的调度和配置优化问题
- 提出了 goodput 的优化目标（在满足 SLA 的前提下最大化吞吐量）
- 量化了 P/D 分离相对于混合部署的提升

### 6.3 vLLM PD 分离

vLLM 从 0.6.x 版本开始支持实验性的 PD 分离（通过 `--enable-disagg-prefill`）。

### 6.4 SGLang PD 分离

SGLang 的 PD 分离支持更为完善，配合 RDMA 可以实现低延迟的 KV Cache 传输。

---

## 七、优缺点分析

### 7.1 优点

1. **消除 Prefill-Decode 干扰**：各自独立调度，Decode 不被 Prefill 打断
2. **降低 P99 延迟**：Decode 延迟更稳定，尾延迟大幅降低
3. **提高总体吞吐量**：P 节点和 D 节点各自优化，整体效率更高
4. **支持异构部署**：可以为不同阶段选择最适合的硬件
5. **独立扩容**：根据瓶颈独立扩展 P 或 D 节点

### 7.2 缺点

1. **KV Cache 传输开销**：增加了 TTFT（需要传输 KV Cache）
2. **系统复杂性增加**：需要额外的路由、调度、传输层
3. **资源利用率可能降低**：P 节点在 Decode 阶段空闲
4. **网络要求高**：需要 RDMA 等高带宽低延迟网络
5. **KV Cache 管理复杂**：KV Cache 分布在不同节点

---

## 八、总结

PD 分离是解决 LLM 推理中 Prefill 和 Decode 特性差异问题的核心架构创新：

1. **本质**：匹配计算密集型（Prefill）和内存密集型（Decode）与对应的硬件
2. **核心挑战**：KV Cache 的高效传输（需要 RDMA 等高速网络）
3. **主要收益**：消除干扰、降低尾延迟、提高吞吐量
4. **适用场景**：大规模 LLM 服务（QPS 高、延迟要求严格的场景）
5. **发展趋势**：结合 KV Cache 多级缓存、弹性扩缩容，成为大型推理集群的标准架构

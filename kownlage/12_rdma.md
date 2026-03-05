# 12. RDMA（Remote Direct Memory Access）详解

## 一、概念：远程直接内存访问

### 1.1 什么是 RDMA

RDMA（Remote Direct Memory Access，远程直接内存访问）是一种**绕过 CPU 和操作系统内核**，允许一台计算机直接访问另一台计算机内存的网络通信技术。

```
传统网络通信（TCP/IP）：
  Application → System Call → OS Kernel → TCP/IP Stack → Network Driver → NIC → 网络
  （每一步都需要 CPU 参与，涉及多次内存拷贝）

  发送端: App Buffer → Kernel Buffer → NIC Buffer → 网络
  接收端: 网络 → NIC Buffer → Kernel Buffer → App Buffer
  = 4 次内存拷贝 + 多次 CPU 中断 + 上下文切换

RDMA：
  Application → RDMA Verbs → NIC (RNIC) → 网络 → 远端 NIC → 远端内存
  （绕过 OS 内核，DMA 直接操作内存，CPU 几乎不参与）

  发送端: App Buffer → 网络（NIC 直接从用户空间 DMA 读取）
  接收端: 网络 → App Buffer（NIC 直接 DMA 写入用户空间）
  = 0 次 CPU 拷贝，零内核参与
```

### 1.2 RDMA 的三大核心特性

1. **零拷贝（Zero Copy）**
   - 数据直接从应用程序缓冲区发送到网络，不经过内核缓冲区
   - 接收端同样直接写入应用程序缓冲区
   - 消除了内核态↔用户态之间的数据拷贝

2. **内核旁路（Kernel Bypass）**
   - 数据路径完全在用户态完成，不涉及系统调用
   - 不经过 TCP/IP 协议栈
   - 不触发 CPU 中断

3. **CPU Offload**
   - 协议处理、数据传输都由 RNIC（RDMA-capable NIC）硬件完成
   - CPU 只需要发起请求，不参与实际数据搬运
   - CPU 可以在数据传输期间做其他事情

### 1.3 RDMA 操作类型

```
1. RDMA Send/Receive（双边操作）
   - 类似传统的 send/recv
   - 接收端需要预先 post receive buffer
   - 双方都需要参与

2. RDMA Write（单边操作）★ 最常用
   - 发送端直接写入接收端的内存
   - 接收端 CPU 完全不感知（甚至不知道数据已经到了！）
   - 需要事先知道接收端的内存地址和 rkey（remote key）

3. RDMA Read（单边操作）
   - 发送端直接从接收端的内存读取数据
   - 接收端 CPU 同样不感知

4. RDMA Atomic（原子操作）
   - 支持远程原子操作：Compare-and-Swap、Fetch-and-Add
   - 用于分布式锁、计数器等
```

**RDMA Write 是分布式推理中最关键的操作**——一个节点可以直接将 tensor 数据写入另一个节点的 GPU 显存，接收端的 CPU/GPU 完全不需要参与。

---

## 二、核心技术

### 2.1 InfiniBand（IB）

InfiniBand 是一种**专门为 RDMA 设计的网络技术**，提供端到端的 RDMA 原生支持。

```
InfiniBand 网络架构：
  Server 1 (HCA) ──── IB Switch ──── Server 2 (HCA)
                       │
                  IB Switch (可级联)
                       │
                  Server 3 (HCA)

组件：
- HCA（Host Channel Adapter）：InfiniBand 网卡
- IB Switch：InfiniBand 交换机
- IB Router：跨子网连接
```

**InfiniBand 速率演进：**

| 版本 | 单通道速率 | 4x 速率 | 年代 |
|------|-----------|---------|------|
| SDR | 2.5 Gbps | 10 Gbps | 2000s |
| DDR | 5 Gbps | 20 Gbps | 2000s |
| QDR | 10 Gbps | 40 Gbps | 2010s |
| FDR | 14 Gbps | 56 Gbps | 2010s |
| EDR | 25 Gbps | 100 Gbps | 2016 |
| HDR | 50 Gbps | 200 Gbps | 2019 |
| NDR | 100 Gbps | 400 Gbps | 2022 |
| XDR | 200 Gbps | 800 Gbps | 2024 |

**当前主流 AI 集群使用 NDR（400Gbps）或 HDR（200Gbps）InfiniBand。**

**InfiniBand 的特点：**
- 专用网络，需要专用交换机和网卡
- 端到端延迟极低：**~1 微秒**
- 原生支持 RDMA，无需额外协议封装
- 主要厂商：NVIDIA/Mellanox（已被 NVIDIA 收购）
- 成本较高，主要用于 HPC 和 AI 数据中心

### 2.2 RoCE v2（RDMA over Converged Ethernet）

RoCE v2 是在**以太网**上实现 RDMA 的技术。

```
协议栈对比：

InfiniBand:
  RDMA Verbs → IB Transport → IB Link Layer → IB Physical

RoCE v1:
  RDMA Verbs → IB Transport → Ethernet Link Layer → Ethernet Physical
  （仅支持同一个二层网络，不能跨路由器）

RoCE v2:
  RDMA Verbs → IB Transport → UDP/IP → Ethernet Link Layer → Ethernet Physical
  （支持三层路由，可以跨网段）★ 主流方案
```

**RoCE v2 的特点：**
- 利用现有以太网基础设施，成本远低于 InfiniBand
- 需要特殊的以太网交换机支持（支持 ECN、PFC 等流控）
- 需要 **无损以太网（Lossless Ethernet）** 或 **DCQCN 拥塞控制**
- 延迟略高于 InfiniBand：**~2-5 微秒**
- 适合预算有限但需要 RDMA 的场景

**RoCE v2 与 InfiniBand 对比：**

| 特性 | InfiniBand | RoCE v2 |
|------|-----------|---------|
| 延迟 | ~1 μs | ~2-5 μs |
| 带宽 | 最高 800 Gbps | 最高 400 Gbps |
| 网络基础设施 | 专用 | 基于以太网 |
| 成本 | 高 | 中等 |
| 拥塞控制 | 原生（credit-based）| 需要 DCQCN/ECN |
| 可靠性 | 高（原生无损）| 需要额外配置 PFC |
| 适用场景 | 大型 AI 集群 | 中小型集群/企业 |

### 2.3 iWARP（Internet Wide Area RDMA Protocol）

另一种在以太网上实现 RDMA 的协议，基于 TCP：
- 优点：可以在有损网络上运行（TCP 保证可靠性）
- 缺点：性能不如 RoCE v2，延迟较高
- 市场占有率低，不太常用于 AI 场景

---

## 三、在分布式推理中的应用

### 3.1 跨节点 Tensor 传输

在多节点分布式推理中，RDMA 用于节点间的 tensor 数据传输：

```
场景：Tensor Parallelism 跨两个节点

Node 1 (GPU 0-3)                    Node 2 (GPU 4-7)
┌─────────────────┐                 ┌─────────────────┐
│ Layer N 计算完成  │                 │ 等待 AllReduce    │
│ Partial Result   │──── RDMA ────→ │ Partial Result   │
│                  │←── RDMA ────── │                  │
│ AllReduce 完成   │                 │ AllReduce 完成   │
│ Layer N+1 开始   │                 │ Layer N+1 开始   │
└─────────────────┘                 └─────────────────┘

传统 TCP/IP: AllReduce 通信延迟可能是 ~100-500 μs
RDMA:       AllReduce 通信延迟可以降到 ~10-50 μs
```

**NCCL（NVIDIA Collective Communications Library）底层就使用 RDMA**：
- 当检测到 InfiniBand 或 RoCE 时，NCCL 自动使用 RDMA
- NCCL 对用户透明，开发者只需调用 AllReduce/AllGather 等集合通信原语
- NCCL 会自动选择最优的传输方式

### 3.2 KV Cache 跨节点传输（PD 分离架构）

在 **Prefill-Decode 分离（PD 分离）** 架构中，RDMA 是 KV Cache 迁移的关键：

```
PD 分离架构中的 KV Cache 传输：

Prefill Node                              Decode Node
┌──────────────────┐                     ┌──────────────────┐
│                  │                     │                  │
│  执行 Prefill    │                     │                  │
│  ↓               │                     │                  │
│  KV Cache 生成   │                     │                  │
│  ↓               │                     │                  │
│  RDMA Write  ────┼──── 直接写入 ──────→│  KV Cache 到达   │
│  (GPU→GPU)       │    接收端显存       │  ↓               │
│                  │                     │  开始 Decode     │
│                  │                     │  逐 token 生成   │
└──────────────────┘                     └──────────────────┘

KV Cache 大小（DeepSeek-V3 为例）：
  - 61 层 × (512 维 compressed KV) × seq_len × dtype_size
  - 4096 tokens 的请求 ≈ 几十 MB 到 几百 MB

传输延迟对比：
  TCP/IP (100Gbps):  100MB / 12.5GB/s ≈ 8 ms + 协议开销 ≈ 10-15 ms
  RDMA (400Gbps IB): 100MB / 50GB/s ≈ 2 ms + 极少开销 ≈ 2-3 ms

  对于实时推理，这 10ms 的差距非常关键！
```

### 3.3 分布式 KV Cache 共享

在大规模部署中，多个推理实例可能需要**共享 KV Cache**（如 prefix caching 跨实例共享）：

```
Instance 1 (System Prompt KV Cache)
       │
       │ RDMA Read (单边操作)
       │ 直接从 Instance 1 显存读取，Instance 1 的 CPU/GPU 无感知
       ▼
Instance 2 (获取共享的 KV Cache)

优势：
  - 单边操作，不打扰数据源节点的计算
  - 零拷贝，数据直接进入目标 GPU 显存
  - 极低延迟
```

---

## 四、与 TCP/IP 的性能对比

### 4.1 延迟对比

```
操作          | TCP/IP (100GbE)  | RDMA (InfiniBand NDR)
---------------------------------------------------------
小消息延迟     | 10-50 μs         | 1-2 μs
大消息延迟     | 50-500 μs        | 5-20 μs (取决于大小)
CPU 开销       | 高（协议栈处理）   | 极低（硬件卸载）
内存拷贝次数   | 2-4 次            | 0 次（零拷贝）
上下文切换     | 多次               | 0 次
```

### 4.2 带宽对比

```
理论带宽：
  100GbE TCP:        ~12.5 GB/s（协议开销后有效带宽更低）
  200Gbps HDR IB:    ~25 GB/s
  400Gbps NDR IB:    ~50 GB/s
  800Gbps XDR IB:    ~100 GB/s

实际有效带宽（大消息场景）：
  TCP/IP:  理论带宽的 70-85%（协议开销、内核处理）
  RDMA:    理论带宽的 90-98%（硬件直接传输）
```

### 4.3 CPU 开销对比

```
TCP/IP:
  发送 1GB 数据 → CPU 需要处理 ~700K 个 TCP 段
  → 每个段：校验和计算、序列号管理、拥塞控制、内存拷贝
  → 占用一个 CPU 核约 30-50% 的计算能力

RDMA:
  发送 1GB 数据 → CPU 只需发起一次 RDMA Write 操作
  → 所有传输由 NIC 硬件完成
  → CPU 占用 < 1%
  → CPU 可以继续做有用的计算工作
```

### 4.4 对推理延迟的影响

```
场景：DeepSeek-V3 的 Tensor Parallelism AllReduce

每步推理需要做多次 AllReduce：
  - 每层 2 次 AllReduce（attention 后 + FFN 后）
  - 61 层 = 122 次 AllReduce
  - 每次传输几 MB 的数据

TCP/IP (假设每次 AllReduce 50μs):
  122 × 50μs = 6.1 ms 的通信开销

RDMA (假设每次 AllReduce 5μs):
  122 × 5μs = 0.61 ms 的通信开销

差距：5.5 ms per token，这在实时推理中是不可接受的延迟！

更关键的是：TCP/IP 的 CPU 开销会与 GPU kernel launch 竞争 CPU 资源，
进一步拖慢推理速度。RDMA 几乎不占 CPU，完全避免了这个问题。
```

---

## 五、GPUDirect RDMA

### 5.1 传统 GPU 跨节点通信（无 GPUDirect RDMA）

```
Node 1 GPU → Node 2 GPU 传输路径：

1. GPU 显存 → Host 内存（PCIe DMA，cudaMemcpy D2H）
2. Host 内存 → NIC 缓冲区（CPU 参与的内存拷贝或 DMA）
3. NIC → 网络 → 远端 NIC
4. 远端 NIC → 远端 Host 内存
5. 远端 Host 内存 → 远端 GPU 显存（PCIe DMA，cudaMemcpy H2D）

总共 4 次 PCIe 穿越 + 2 次 Host 内存拷贝
延迟高，带宽受限于多个瓶颈点
```

### 5.2 GPUDirect RDMA（GDR）

```
GPUDirect RDMA 的传输路径：

1. GPU 显存 → NIC（通过 PCIe 直接 DMA，绕过 Host 内存！）
2. NIC → 网络 → 远端 NIC
3. 远端 NIC → 远端 GPU 显存（通过 PCIe 直接 DMA）

总共 2 次 PCIe 穿越，0 次 Host 内存拷贝
延迟大幅降低，带宽大幅提升
```

```
对比图：

Without GPUDirect RDMA:
  GPU ──PCIe──> CPU/Mem ──PCIe──> NIC ═══network═══ NIC ──PCIe──> CPU/Mem ──PCIe──> GPU

With GPUDirect RDMA:
  GPU ──PCIe──> NIC ═══════network═══════ NIC ──PCIe──> GPU
  (CPU 和 Host 内存完全不参与！)
```

### 5.3 GPUDirect RDMA 的硬件要求

- **NVIDIA GPU**：支持 GPUDirect RDMA（Kepler 及以后的架构，即 GPU >= K40）
- **RDMA NIC**：Mellanox/NVIDIA ConnectX 系列网卡
- **PCIe 拓扑**：GPU 和 NIC 最好在同一个 PCIe switch 下（否则需要穿越 CPU 的 QPI/UPI）
- **nvidia-peermem 内核模块**：允许 NIC 直接访问 GPU 显存

### 5.4 PCIe 拓扑对性能的影响

```
最优拓扑（GPU 和 NIC 在同一 PCIe switch 下）：
  GPU ──PCIe──> PCIe Switch ──PCIe──> NIC
  延迟最低，带宽最高

次优拓扑（GPU 和 NIC 在不同 PCIe switch，需经过 CPU）：
  GPU ──PCIe──> PCIe Switch1 ──UPI──> CPU ──UPI──> PCIe Switch2 ──PCIe──> NIC
  需要穿越 CPU 的 UPI 总线，延迟增加，带宽受 UPI 限制

NVIDIA DGX/HGX 系统专门优化了 PCIe 拓扑，确保 GPU-NIC 亲和性。
```

### 5.5 GPUDirect 技术族谱

```
GPUDirect 技术演进：

1. GPUDirect Peer-to-Peer (P2P): 同一节点内 GPU 间直接通过 PCIe 传输
2. GPUDirect RDMA (GDR): GPU 显存与远端网卡之间直接 DMA
3. GPUDirect Storage: GPU 显存与 NVMe SSD 之间直接 DMA
4. GPUDirect Video: GPU 显存与视频采集设备之间直接 DMA

所有这些技术的核心思想：消除不必要的 CPU/Host 内存中转。
```

---

## 六、NVLink、NVSwitch 与 RDMA 的关系

### 6.1 NVLink

**NVLink** 是 NVIDIA 开发的**高速 GPU 互联技术**，用于**同一节点内** GPU 之间的直接通信。

```
NVLink 与 PCIe 对比：

PCIe Gen5 x16: 单向 ~64 GB/s, 双向 ~128 GB/s
NVLink 4.0:    单向 ~450 GB/s（18 links × 25 GB/s），双向 ~900 GB/s

NVLink 比 PCIe 快 ~7x！
```

**NVLink 的演进：**

| 版本 | 单 link 带宽 | GPU 架构 | 典型配置 |
|------|-------------|---------|---------|
| NVLink 1.0 | 40 GB/s | Pascal (P100) | 4 links |
| NVLink 2.0 | 50 GB/s | Volta (V100) | 6 links |
| NVLink 3.0 | 50 GB/s | Ampere (A100) | 12 links |
| NVLink 4.0 | 50 GB/s | Hopper (H100) | 18 links |
| NVLink 5.0 | 100 GB/s | Blackwell (B200) | 18 links |

### 6.2 NVSwitch

**NVSwitch** 是 NVIDIA 的 **NVLink 交换机芯片**，用于实现节点内**所有 GPU 全互联**。

```
Without NVSwitch（点对点 NVLink）：
  GPU0 ── NVLink ── GPU1
  GPU0 ── NVLink ── GPU2
  GPU0 ── NVLink ── GPU3
  ...
  每个 GPU 的 NVLink 端口有限，无法全互联 8 个 GPU
  某些 GPU 对之间可能需要多跳

With NVSwitch：
  GPU0 ─┐
  GPU1 ─┤
  GPU2 ─┼── NVSwitch ── 全互联
  GPU3 ─┤
  ...   ─┤
  GPU7 ─┘

  任意两个 GPU 之间都有 full bandwidth NVLink 连接
  8 GPU 总互联带宽：~900 GB/s × 8 = ~7.2 TB/s（H100 DGX）
```

**NVSwitch 的关键价值：**
- 实现 **all-to-all full bisection bandwidth**
- AllReduce 等集合通信操作可以以理论最大带宽执行
- NCCL 在 NVSwitch 上可以达到接近理论峰值的性能

### 6.3 三者的关系和协作

```
大模型分布式推理的完整通信层次：

┌─────────────────────────────────────────────────────────────┐
│                    Multi-Node Cluster                        │
│                                                              │
│  Node 1                              Node 2                  │
│  ┌──────────────────────┐            ┌──────────────────────┐│
│  │ GPU0 ── NVLink ── GPU1│            │ GPU4 ── NVLink ── GPU5││
│  │  │    NVSwitch     │  │            │  │    NVSwitch     │  ││
│  │ GPU2 ── NVLink ── GPU3│            │ GPU6 ── NVLink ── GPU7││
│  │         │             │            │         │             ││
│  │        NIC            │            │        NIC            ││
│  └────────┼──────────────┘            └────────┼──────────────┘│
│           │                                    │              │
│           └──── RDMA (InfiniBand) ─────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

通信层次：
1. 同一 GPU 内部：访问自己的显存（最快，~3 TB/s HBM 带宽）
2. 同一节点 GPU 间：NVLink/NVSwitch（~900 GB/s 双向）
3. 跨节点 GPU 间：RDMA/InfiniBand（~50-100 GB/s）
4. 跨节点 CPU 间：TCP/IP 或 RDMA（最后手段）

速度关系：
  HBM > NVLink >> RDMA >> TCP/IP
  3TB/s > 900GB/s >> 50GB/s >> 12.5GB/s
```

### 6.4 NCCL 如何统一管理这些通信层次

```
NCCL（NVIDIA Collective Communications Library）作为统一的抽象层：

应用层：  torch.distributed.all_reduce(tensor)
            │
NCCL 层：  自动检测硬件拓扑
            │
            ├── GPU 在同一节点且有 NVLink → 使用 NVLink 直接传输
            ├── GPU 在同一节点无 NVLink → 使用 PCIe P2P 或经过 Host
            ├── GPU 在不同节点，有 InfiniBand → 使用 RDMA (GPUDirect RDMA)
            ├── GPU 在不同节点，有 RoCE → 使用 RoCE RDMA
            └── 只有 TCP/IP → 使用 TCP Socket

NCCL 的 Ring AllReduce / Tree AllReduce 算法会：
- 优先使用快速链路（NVLink）
- 跨节点通信走 RDMA
- 自动构建最优的通信拓扑
```

### 6.5 实际部署中的拓扑优化

```
典型 AI 集群部署（如 DGX H100 集群）：

每个节点（DGX H100）：
  - 8x H100 GPU，NVSwitch 全互联（900 GB/s per GPU）
  - 8x ConnectX-7 400Gbps InfiniBand NIC（每个 GPU 一个 NIC）
  - GPU-NIC 亲和性优化：GPU[i] 与 NIC[i] 在同一 PCIe switch 下

跨节点通信：
  - 每个 GPU 有自己专属的 400Gbps NIC
  - 8 个 GPU = 8 × 400Gbps = 3200Gbps = 400 GB/s 总跨节点带宽

Tensor Parallelism（TP）策略：
  - TP 通常部署在同一节点内（利用 NVLink 高带宽）
  - TP=8 时，8 个 GPU 通过 NVSwitch 通信

Pipeline Parallelism（PP）策略：
  - PP 通常跨节点部署
  - 通过 RDMA 传输 activation 数据
  - PP 的通信量远小于 TP，所以可以容忍较低带宽

Expert Parallelism（EP，MoE 模型）：
  - 跨节点的 expert 路由需要 all-to-all 通信
  - 对跨节点带宽需求最大的场景之一
  - RDMA 的高带宽和低延迟在此尤为关键
```

### 6.6 跨节点 NVLink（NVLink Network / NVLink Switch）

NVIDIA 在 GB200 NVL72 等新平台中引入了**跨节点 NVLink**：

```
传统架构：
  节点内 NVLink（~900 GB/s） >> 节点间 RDMA（~50-100 GB/s）
  带宽断崖式下降！

GB200 NVL72 架构：
  72 个 GPU 通过 NVLink Switch 全互联
  即使跨节点，也使用 NVLink 带宽
  彻底消除了节点间带宽瓶颈

这意味着：
  TP 可以跨节点（因为跨节点也有 NVLink 带宽）
  更灵活的并行策略
  更大的模型可以高效推理
```

---

## 七、面试要点总结

1. **RDMA 三大特性**：零拷贝、内核旁路、CPU 卸载——绕过 CPU 和 OS 内核，NIC 硬件直接操作内存
2. **三种 RDMA 实现**：InfiniBand（最强性能）、RoCE v2（基于以太网，性价比高）、iWARP（基于 TCP，不太常用）
3. **RDMA 在推理中的角色**：跨节点 AllReduce、KV Cache 迁移（PD 分离）、分布式 KV Cache 共享
4. **GPUDirect RDMA**：GPU 显存直接通过 NIC 跨节点传输，绕过 Host 内存，延迟减半
5. **通信层次**：HBM（3TB/s）> NVLink（900GB/s）>> RDMA（50-100GB/s）>> TCP/IP（12.5GB/s）
6. **NVLink vs RDMA**：NVLink 用于节点内 GPU 互联，RDMA 用于节点间通信；NCCL 统一管理
7. **实际部署策略**：TP 放节点内（用 NVLink），PP/EP 跨节点（用 RDMA）

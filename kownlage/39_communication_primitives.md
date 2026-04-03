# 分布式通信原语完全指南

> 以 4 张 GPU（编号 0-3）为例，数据块用大写字母表示。

---

## 一、基础问题

### 1.1 卡间链路：单向还是双向？

**双向（Full-Duplex）。**

- **NVLink**：每条 NVLink 都是双向的，例如 NVLink 3.0 单条链路 **每个方向 25 GB/s**，双向合计 50 GB/s。
- **PCIe**：同样是双向全双工，PCIe 4.0 x16 每个方向约 32 GB/s。
- **网络（RDMA/RoCE/InfiniBand）**：也是双向全双工。

这意味着 GPU0→GPU1 发数据的同时，GPU1→GPU0 也能以全带宽发数据，互不干扰。
Ring AllReduce 正是利用了这一点：数据在环中同时双向流动。

**互相传时，单向能打满吗？能！不是对半分。**

```
全双工 (Full-Duplex) — 实际的物理实现:
  A→B 和 B→A 使用不同的物理线路（差分对），互不干扰
  ┌──────────────────┐
  │  A ═══TX线路═══→ B │  带宽 n（独占）
  │  A ←═══RX线路═══ B │  带宽 n（独占）
  └──────────────────┘
  双向同时传时: A→B = n, B→A = n, 不是各 n/2

半双工 (Half-Duplex) — 对比:
  A 和 B 共用同一条线路，轮流使用
  同时传时确实只能各 n/2（但现代互联几乎不存在这种情况）

厂商标注惯例:
  "NVLink 3.0 单链路 50 GB/s" → 通常指双向合计，单向 25 GB/s
  "NVLink 4.0 单链路 100 GB/s" → 双向合计，单向 50 GB/s
  具体看文档里是 "unidirectional" 还是 "bidirectional"
```

**全双工的物理实现 — 就是两组独立的线：**

```
━━━ NVLink 物理层 ━━━

一条 NVLink "链路"内部:

  GPU_A                              GPU_B
  ┌────┐   TX 差分对 (8对线) ────→   ┌────┐
  │    │                              │    │
  │    │   ←──── RX 差分对 (8对线)   │    │
  └────┘                              └────┘

  TX(发送) 和 RX(接收) 是完全独立的物理铜线
  A 的 TX 连到 B 的 RX，A 的 RX 连到 B 的 TX
  所以 A→B 和 B→A 同时满速，因为走的是不同的物理线

  NVLink 3.0 单子链路: 8 个差分对 × 每对 ~3.2 GB/s = ~25 GB/s/方向
  物理上共 32 根线（TX 16 根 + RX 16 根）

━━━ PCIe 物理层 ━━━

  PCIe x16 = 16 条 lane，每条 lane 独立的 TX + RX 差分对
  PCIe 4.0 x16: 每方向 ~32 GB/s，双向合计 ~64 GB/s

━━━ InfiniBand ━━━

  同理，独立 TX/RX 对。HDR 每方向 25 GB/s

差分对 (Differential Pair):
  两根紧挨的铜线传互补信号，接收端取差值抵消共模噪声
  所有高速串行总线 (PCIe/NVLink/USB3/IB) 都用差分对
```

### 1.15 为什么 Ring 带宽最优但延迟最高？

**延迟和带宽是两个独立的物理量，不能混为一谈：**

```
发送一条消息的耗时 = α + M / BW

α  = 启动延迟: 与数据大小无关的固定开销
     (软件栈、DMA 建立、同步握手等，典型 ~1-5μs)
BW = 带宽: 管道有多粗（每秒能灌多少数据）
M  = 消息大小
```

**Ring vs Tree 耗时拆解 (N 节点 AllReduce)：**

```
Ring:  总耗时 = 2(N-1) × α     +  2(N-1)/N × M/BW
                ~~~~~~~~~~~~       ~~~~~~~~~~~~~~~~~~
                延迟项(大)           带宽项(最优，趋近 2M/BW)

Tree:  总耗时 = 2·log₂N × α    +  2·log₂N × M/BW
                ~~~~~~~~~~~~       ~~~~~~~~~~~~~~~~~~
                延迟项(小)           带宽项(较大，带宽利用率~50%)
```

**具体数字 (N=128)：**

```
                   Ring                    Tree
延迟项:    2×127 × 1μs = 254μs     2×7 × 1μs = 14μs

── M = 1KB (小消息) ──
带宽项:    ~0.04μs                   ~0.004μs
总耗时:    ~254μs                    ~14μs        ← Tree 快 18 倍!
           延迟项完全主导

── M = 1GB (梯度同步) ──
带宽项:    ~40ms (带宽利用率100%)     ~80ms (利用率~50%)
总耗时:    ~40ms                     ~80ms        ← Ring 快 2 倍!
           带宽项完全主导，延迟可忽略
```

**一句话: 延迟看"串行等几轮握手"，带宽看"管道塞满没"。
Ring 轮数多(延迟高)但每轮管道都塞满(带宽最优)，大消息场景下带宽项碾压延迟项。**

### 1.2 是否所有原语都用 Ring？

**不是。** 不同原语根据消息大小选择不同拓扑：

| 拓扑 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **Ring** | 大消息（MB 级以上，如梯度同步） | 带宽最优 | 延迟 O(N) |
| **Tree / Recursive Halving-Doubling** | 小消息（KB 级，如 scalar 同步） | 延迟 O(log N) | 带宽利用率不满 |
| **Direct P2P（Send/Recv）** | 点对点传输、Pipeline 并行 | 最简单、延迟最低 | 不适合集合通信 |
| **Butterfly / Recursive Doubling** | 中等消息的 AllReduce | 折中 | 实现复杂 |

NCCL 等库会根据消息大小和节点数**自动选择**最优算法。

### 1.3 Tree 和 Recursive Halving-Doubling 详解

除了 Ring 和 P2P，另外两种重要拓扑的详细过程：

**Tree（树形）— 以 8 节点 Reduce 为例：**

```
轮次1 (叶子→父节点，4 对并行):
  GPU1→GPU0 发数据,  GPU3→GPU2 发数据
  GPU5→GPU4 发数据,  GPU7→GPU6 发数据
  → GPU0 算 0+1, GPU2 算 2+3, GPU4 算 4+5, GPU6 算 6+7

轮次2 (第二层, 2 对并行):
  GPU2→GPU0 发(2+3),  GPU6→GPU4 发(6+7)
  → GPU0 算 0+1+2+3, GPU4 算 4+5+6+7

轮次3 (根节点):
  GPU4→GPU0 发(4+5+6+7)
  → GPU0 算 0+1+2+3+4+5+6+7 ✓

树形结构:
          GPU0 (root)
         /          \
      GPU0          GPU4
      /   \         /   \
   GPU0  GPU2   GPU4  GPU6
   / \    / \    / \    / \
  0   1  2   3  4   5  6   7
```

- 轮数: **log₂(N)** = 3（Ring 需要 7 轮）
- 但每轮只有约一半节点在通信，**带宽利用率 ~50%**
- 适合小消息：延迟项 `log₂(N) * α` 远小于 Ring 的 `(N-1) * α`

**Recursive Halving-Doubling — 以 8 节点 AllReduce 为例：**

```
核心思想: 每轮配对距离翻倍/减半，所有节点同时参与

═══ 阶段一: Recursive Halving (ReduceScatter 效果) ═══

轮次1: 距离=4, 配对 (0↔4)(1↔5)(2↔6)(3↔7)
  每对交换各自数据的一半并归约
  → 节点 0-3 持有前半数据的部分归约
  → 节点 4-7 持有后半数据的部分归约

轮次2: 距离=2, 配对 (0↔2)(1↔3)(4↔6)(5↔7)
  在各自的半区内再对半交换
  → 每个节点持有 1/4 数据的部分归约

轮次3: 距离=1, 配对 (0↔1)(2↔3)(4↔5)(6↔7)
  → 每个节点持有 1/8 数据的完整归约 ✓

═══ 阶段二: Recursive Doubling (AllGather 效果) ═══

轮次4: 距离=1, (0↔1)(2↔3)(4↔5)(6↔7) 互换归约结果
轮次5: 距离=2, (0↔2)(1↔3)(4↔6)(5↔7) 互换
轮次6: 距离=4, (0↔4)(1↔5)(2↔6)(3↔7) 互换
  → 所有节点拥有完整归约结果 ✓
```

- 总轮数: **2 * log₂(N)** = 6
- **每轮所有节点都在通信**（比 Tree 好）
- 但每轮传输量递减/递增，带宽利用率不如 Ring 满
- 适合中等消息大小

**四种拓扑完整对比：**

| | Ring | Tree | Recursive H-D | P2P |
|---|---|---|---|---|
| 轮数 | N-1 | log₂N | 2·log₂N | 1 |
| 带宽效率 | **最优 ~100%** | ~50% | ~60-80% | N/A |
| 延迟 | O(N)·α | **O(logN)·α** | O(logN)·α | **O(1)·α** |
| 适合消息大小 | 大 (>256KB) | 小 (<数KB) | 中等 | 任意(点对点) |
| 节点利用率 | 所有节点每轮都参与 | 每轮约半数空闲 | 所有节点每轮都参与 | 仅两个节点 |

### 1.4 同一原语，NCCL 会选择不同拓扑吗？

**是的，这是 NCCL 的核心设计。** 同一个 `ncclAllReduce` 调用，内部根据多种因素选不同算法：

```
ncclAllReduce 内部决策逻辑:

1. 消息大小 (最主要因素)
   ├── < 数KB     → Tree (延迟优先)
   ├── 数KB ~ 数MB → Recursive Halving-Doubling 或小型 Ring
   └── > 数MB     → Ring (带宽优先)

2. 节点数
   ├── 2 节点    → 直接 P2P 交换 (最简单)
   ├── 少量节点  → Tree 可能更优
   └── 大量节点  → Ring (延迟项虽然大但带宽优势更明显)

3. 物理拓扑探测 (NCCL 初始化时自动探测)
   ├── 节点内 NVLink/NVSwitch → NVLink 优化的 Tree
   ├── 节点间 InfiniBand      → Ring 或 SHARP
   └── 混合环境 → 分层策略: 节点内一种 + 节点间一种

4. 特殊硬件加速
   ├── InfiniBand SHARP → 网络交换机内直接做归约
   └── NVSwitch NVLS    → 利用 NVSwitch 做 multicast/reduction

环境变量可手动覆盖:
  NCCL_ALGO=Ring|Tree|CollnetDirect|CollnetChain|NVLS
  NCCL_PROTO=Simple|LL|LL128
  NCCL_MIN_NCHANNELS / NCCL_MAX_NCHANNELS  通道数
```

---

## 二、点对点原语

### 2.1 Send / Recv（发送/接收）

最基础的通信原语，一对一，指定源和目标。

```
GPU0 ──Send(A)──→ GPU2

GPU0: [A]  →  GPU0: [A]
GPU1: [ ]     GPU1: [ ]
GPU2: [ ]     GPU2: [A]    ← Recv
GPU3: [ ]     GPU3: [ ]
```

**用途**：Pipeline 并行中，stage 之间传递 activation / gradient。
**拓扑**：纯 P2P，不涉及 Ring/Tree。

---

## 三、集合通信原语（Collective Operations）

### 3.1 Broadcast（广播）

**一个节点的数据复制到所有节点。**

```
root=GPU0, 数据=A

初始:                广播后:
GPU0: [A]           GPU0: [A]
GPU1: [ ]    →      GPU1: [A]
GPU2: [ ]           GPU2: [A]
GPU3: [ ]           GPU3: [A]
```

**Ring 实现（3 轮）：**

```
环: 0 → 1 → 2 → 3 → 0

轮次1: GPU0→GPU1 发A
  GPU0:[A]  GPU1:[A]  GPU2:[ ]  GPU3:[ ]

轮次2: GPU1→GPU2 发A
  GPU0:[A]  GPU1:[A]  GPU2:[A]  GPU3:[ ]

轮次3: GPU2→GPU3 发A
  GPU0:[A]  GPU1:[A]  GPU2:[A]  GPU3:[A]
```

**Tree 实现（2 轮，更适合小消息）：**

```
轮次1: GPU0→GPU1, GPU0→GPU2 同时发A
轮次2: GPU2→GPU3 发A
（树形扇出，延迟 O(log N)）
```

---

### 3.2 Scatter（散播）

**root 节点将数据的不同块分发给各节点，每个节点得到一块。**

```
root=GPU0, 数据被分成4块: A B C D

初始:                     散播后:
GPU0: [A B C D]          GPU0: [A]
GPU1: [       ]    →     GPU1: [B]
GPU2: [       ]          GPU2: [C]
GPU3: [       ]          GPU3: [D]
```

**Ring 实现（3 轮）：**

```
环: 0 → 3 → 2 → 1 (反向更高效)

轮次1: GPU0→GPU3 发 D,B,C（GPU3之后的节点需要的块）
  实际上 GPU0→GPU3 发 {B,C,D}

更准确的 Ring Scatter:
轮次1: GPU0→GPU3 发 [B,C,D]
轮次2: GPU3→GPU2 发 [B,C]    GPU3 保留 D
轮次3: GPU2→GPU1 发 [B]      GPU2 保留 C
                              GPU1 保留 B
```

> 注意：Scatter 和 Gather 因为数据集中在一个节点，即使用 Ring 也避免不了 root 的带宽瓶颈。
> 实际中更常用的是 ReduceScatter 和 AllGather（所有节点均匀参与）。

---

### 3.3 Gather（收集）

**Scatter 的逆操作：每个节点发送自己的数据块，root 收集所有块。**

```
root=GPU0

初始:                     收集后:
GPU0: [A]                GPU0: [A B C D]
GPU1: [B]        →       GPU1: [B]
GPU2: [C]                GPU2: [C]
GPU3: [D]                GPU3: [D]
```

**Ring 实现（3 轮）：**

```
环: 1 → 2 → 3 → 0

轮次1: GPU1→GPU2 发 B
  GPU2 现在有 [B, C]

轮次2: GPU2→GPU3 发 [B, C]
  GPU3 现在有 [B, C, D]

轮次3: GPU3→GPU0 发 [B, C, D]
  GPU0 现在有 [A, B, C, D] ✓
```

**要点**：是的，Ring Gather 需要 N-1 轮，数据在环上逐步累积流向 root。
这不是 P2P 直发，而是流水线式传递，每一跳附带之前的数据。

---

### 3.4 Reduce（归约）

**所有节点的数据做归约运算（如 sum），结果汇聚到 root。**

```
root=GPU0, op=SUM

初始:                     归约后:
GPU0: [A]                GPU0: [A+B+C+D]
GPU1: [B]        →       GPU1: [B]
GPU2: [C]                GPU2: [C]
GPU3: [D]                GPU3: [D]
```

**Ring 实现（3 轮）：**

```
环: 3 → 2 → 1 → 0

轮次1: GPU3→GPU2 发 D
  GPU2 计算 C+D

轮次2: GPU2→GPU1 发 C+D
  GPU1 计算 B+C+D

轮次3: GPU1→GPU0 发 B+C+D
  GPU0 计算 A+B+C+D ✓
```

**Tree 实现（2 轮，小消息更优）：**

```
轮次1: GPU1→GPU0 发B, GPU3→GPU2 发D  (并行)
  GPU0 算 A+B, GPU2 算 C+D

轮次2: GPU2→GPU0 发 C+D
  GPU0 算 A+B+C+D ✓
```

---

### 3.5 AllGather（全收集）

**每个节点都有一块数据，结束后所有节点都拥有完整数据。**

```
初始:                     全收集后:
GPU0: [A]                GPU0: [A B C D]
GPU1: [B]        →       GPU1: [A B C D]
GPU2: [C]                GPU2: [A B C D]
GPU3: [D]                GPU3: [A B C D]
```

**Ring 实现（3 轮）— 步步详解：**

```
环: 0 → 1 → 2 → 3 → 0 (同时所有节点向右邻居发数据)

═══ 初始状态 ═══
GPU0: [A]    GPU1: [B]    GPU2: [C]    GPU3: [D]

═══ 轮次 1: 每个节点把自己持有的块发给右邻居 ═══
  GPU0→GPU1 发 A
  GPU1→GPU2 发 B
  GPU2→GPU3 发 C
  GPU3→GPU0 发 D     ← 所有传输同时发生！

结果:
  GPU0: [A, D]    GPU1: [A, B]    GPU2: [B, C]    GPU3: [C, D]

═══ 轮次 2: 每个节点把上一轮收到的块继续转发给右邻居 ═══
  GPU0→GPU1 发 D
  GPU1→GPU2 发 A
  GPU2→GPU3 发 B
  GPU3→GPU0 发 C

结果:
  GPU0: [A, C, D]    GPU1: [A, B, D]    GPU2: [A, B, C]    GPU3: [B, C, D]

═══ 轮次 3: 继续转发 ═══
  GPU0→GPU1 发 C
  GPU1→GPU2 发 D
  GPU2→GPU3 发 A
  GPU3→GPU0 发 B

结果:
  GPU0: [A, B, C, D] ✓   GPU1: [A, B, C, D] ✓
  GPU2: [A, B, C, D] ✓   GPU3: [A, B, C, D] ✓
```

**关键**：
- 每轮所有节点**同时收发**，每条链路传输 M/N 数据
- N-1 轮完成，但带宽利用率 100%
- 总通信量每节点: `(N-1) * M/N`

---

### 3.6 ReduceScatter（归约散播）

**所有节点的数据做归约，但结果被分块，每个节点只得到一块归约结果。**

```
op=SUM, 每个节点有 4 块数据

初始:                           归约散播后:
GPU0: [a0 a1 a2 a3]           GPU0: [a0+b0+c0+d0]
GPU1: [b0 b1 b2 b3]    →      GPU1: [a1+b1+c1+d1]
GPU2: [c0 c1 c2 c3]           GPU2: [a2+b2+c2+d2]
GPU3: [d0 d1 d2 d3]           GPU3: [a3+b3+c3+d3]
```

**Ring 实现（3 轮）— 步步详解：**

```
环: 0 → 1 → 2 → 3 → 0

目标: GPU0 负责块0, GPU1 负责块1, GPU2 负责块2, GPU3 负责块3

═══ 初始状态 ═══
GPU0: [a0, a1, a2, a3]
GPU1: [b0, b1, b2, b3]
GPU2: [c0, c1, c2, c3]
GPU3: [d0, d1, d2, d3]

═══ 轮次 1: 每个节点把「左邻居负责的块的前一个」发给右邻居 ═══
  GPU0→GPU1 发 a3    (块3 是 GPU3 负责的，但数据在环上流动)
  GPU1→GPU2 发 b0
  GPU2→GPU3 发 c1
  GPU3→GPU0 发 d2

  接收方做归约:
  GPU1: 块3 部分累加 = a3+b3
  GPU2: 块0 部分累加 = b0+c0
  GPU3: 块1 部分累加 = c1+d1
  GPU0: 块2 部分累加 = a2+d2

═══ 轮次 2: 继续传递部分归约结果 ═══
  GPU0→GPU1 发 (a2+d2)     → 块2
  GPU1→GPU2 发 (a3+b3)     → 块3
  GPU2→GPU3 发 (b0+c0)     → 块0
  GPU3→GPU0 发 (c1+d1)     → 块1

  接收方继续归约:
  GPU1: 块2 = a2+b2+d2
  GPU2: 块3 = a3+b3+c3
  GPU3: 块0 = b0+c0+d0
  GPU0: 块1 = a1+c1+d1

═══ 轮次 3: 最后一轮 ═══
  GPU0→GPU1 发 (a1+c1+d1)  → 块1
  GPU1→GPU2 发 (a2+b2+d2)  → 块2
  GPU2→GPU3 发 (a3+b3+c3)  → 块3
  GPU3→GPU0 发 (b0+c0+d0)  → 块0

  最终归约:
  GPU0: 块0 = a0+b0+c0+d0 ✓
  GPU1: 块1 = a1+b1+c1+d1 ✓
  GPU2: 块2 = a2+b2+c2+d2 ✓
  GPU3: 块3 = a3+b3+c3+d3 ✓
```

**关键**：每轮每个节点只发送 M/N 数据，同时做局部归约，带宽最优。

---

### 3.7 AllReduce（全归约）

**所有节点的数据做归约，结果所有节点都拥有完整的归约结果。**

```
op=SUM

初始:                     全归约后:
GPU0: [A]                GPU0: [A+B+C+D]
GPU1: [B]        →       GPU1: [A+B+C+D]
GPU2: [C]                GPU2: [A+B+C+D]
GPU3: [D]                GPU3: [A+B+C+D]
```

**AllReduce = ReduceScatter + AllGather（Ring 实现）：**

```
═══ 阶段一: ReduceScatter (3 轮) ═══

  将每个节点的数据分为 4 块后执行 ReduceScatter（过程同 3.6）

  结果:
  GPU0: [SUM块0]    GPU1: [SUM块1]    GPU2: [SUM块2]    GPU3: [SUM块3]
  (每个节点持有完整归约结果的 1/4)

═══ 阶段二: AllGather (3 轮) ═══

  执行 AllGather（过程同 3.5）

  结果:
  GPU0: [SUM块0, SUM块1, SUM块2, SUM块3] ✓
  GPU1: [SUM块0, SUM块1, SUM块2, SUM块3] ✓
  GPU2: [SUM块0, SUM块1, SUM块2, SUM块3] ✓
  GPU3: [SUM块0, SUM块1, SUM块2, SUM块3] ✓

共 2*(N-1) = 6 轮，每轮每节点传输 M/N
总耗时: 2*(N-1) * [α + M/(N*BW)]
```

**为什么不用 Reduce + Broadcast？**

```
Reduce + Broadcast:  root 单点瓶颈，带宽 O(N*M)   集中在一个节点
ReduceScatter + AllGather: 负载均匀，带宽 O(M)    分摊到所有节点

后者带宽项达到理论下界 2*(N-1)/N * M/BW ≈ 2M/BW（N 大时）
```

---

## 四、其他通信原语

### 4.1 AllToAll（全交换）

**每个节点给每个节点发送不同的数据（转置操作）。**

```
初始:                          全交换后:
GPU0: [a0 a1 a2 a3]          GPU0: [a0 b0 c0 d0]
GPU1: [b0 b1 b2 b3]    →     GPU1: [a1 b1 c1 d1]
GPU2: [c0 c1 c2 c3]          GPU2: [a2 b2 c2 d2]
GPU3: [d0 d1 d2 d3]          GPU3: [a3 b3 c3 d3]
```

**用途**：MoE（Mixture of Experts）中 token 重新分配、张量并行的维度转换。

### 4.2 Barrier（屏障同步）

**所有节点在此等待，直到所有节点都到达屏障点。**

```
GPU0: ──work──→ |barrier| ──→ continue
GPU1: ──work────→ |barrier| ──→ continue
GPU2: ──work──────→ |barrier| ──→ continue  (最慢的)
GPU3: ──work───→ |barrier| ──→ continue

所有人等 GPU2 到达后才继续
```

**用途**：确保所有节点完成某阶段后再进入下一阶段。不传数据，只同步控制流。

### 4.3 ReduceScatter + AllGather vs Reduce + Broadcast 总结

```
              Reduce+Broadcast          ReduceScatter+AllGather
            ┌──────────────────┐      ┌──────────────────────────┐
            │   集中式(星形)     │      │     分散式(环形)           │
            │                  │      │                          │
            │     ┌─root─┐    │      │  0──1──2──3──0           │
            │    ╱ │  │  ╲   │      │  均匀分担                  │
            │   1  2  3  ...  │      │                          │
            │   单点瓶颈       │      │  无瓶颈                   │
            └──────────────────┘      └──────────────────────────┘
```

---

## 五、完整通信原语一览表

| 原语 | 输入 | 输出 | 常用拓扑 | 典型用途 |
|------|------|------|---------|---------|
| **Send/Recv** | 1→1 | 点对点 | P2P 直连 | Pipeline 并行 |
| **Broadcast** | 1→All (复制) | 所有节点相同 | Tree/Ring | 分发模型参数 |
| **Scatter** | 1→All (分块) | 每节点不同块 | Tree/Ring | 数据分发 |
| **Gather** | All→1 (拼接) | root 有全部 | Tree/Ring | 数据收集 |
| **Reduce** | All→1 (归约) | root 有归约结果 | Tree/Ring | 汇总统计 |
| **AllGather** | All→All (拼接) | 所有节点有全部 | Ring | 张量并行收集 |
| **ReduceScatter** | All→All (归约+分块) | 每节点有部分归约 | Ring | AllReduce 的一半 |
| **AllReduce** | All→All (归约) | 所有节点有完整归约 | Ring | **梯度同步** |
| **AllToAll** | All→All (转置) | 每节点收到各节点的对应块 | P2P/Ring | MoE, 张量并行 |
| **Barrier** | 无数据 | 同步点 | Tree | 阶段同步 |

---

## 六、拓扑选择决策树

```
消息大小?
├── 小消息 (< ~256KB)
│   └── 用 Tree / Recursive Halving-Doubling
│       延迟 O(log N), 带宽利用率不满但延迟低
│
├── 大消息 (> ~256KB)  ← 深度学习梯度同步通常在这
│   └── 用 Ring
│       延迟 O(N), 但带宽利用率 100%
│
└── 点对点 / Pipeline
    └── 用 Direct P2P (Send/Recv)
        延迟 O(1), 最简单

注: NCCL 会根据消息大小、节点数、拓扑自动选择最优算法
    阈值可通过环境变量 NCCL_ALGO 和 NCCL_PROTO 调整
```

---

## 七、通信库对比

### 7.1 主流通信库一览

| 库 | 维护方 | 目标硬件 | 特点 |
|---|---|---|---|
| **NCCL** | NVIDIA | NVIDIA GPU | GPU 集合通信事实标准，深度优化 NVLink/NVSwitch/IB |
| **Gloo** | Meta | CPU (+GPU) | CPU 通信为主，PyTorch 默认 CPU 后端，支持 TCP/共享内存 |
| **MPI** (OpenMPI/MPICH) | 开源社区 | CPU (+GPU) | HPC 领域标准，最完整的原语集，通用但非 GPU 专用 |
| **RCCL** | AMD | AMD GPU (ROCm) | NCCL 的 AMD 移植，API 完全兼容 |
| **oneCCL** | Intel | Intel CPU/GPU/Gaudi | Intel 硬件优化，支持 Habana Gaudi 加速卡 |
| **MSCCL / MSCCL++** | Microsoft | NVIDIA GPU | 可编程集合通信，用户可自定义拓扑和算法 |
| **BCCL** | 百度(昆仑芯) | 昆仑芯 XPU | 昆仑芯片专用集合通信库 |
| **HCCL** | 华为 | 昇腾 NPU | 昇腾芯片专用集合通信库 |

### 7.2 NCCL vs Gloo 核心区别

```
                    NCCL                           Gloo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标场景        GPU ↔ GPU 高性能通信            CPU ↔ CPU 通信 (也支持GPU)
传输层          NVLink, PCIe, InfiniBand        TCP, 共享内存, (IB)
                GPUDirect RDMA (零拷贝)
实现方式        CUDA kernel 直接操作 GPU 显存    用户态 C++，数据经主机内存中转
GPU性能         最优                             远不如 NCCL (需 GPU→CPU→网络→CPU→GPU)
CPU性能         不支持                           足够好
原语完整度      集合通信 + P2P                   集合通信 + P2P
容错能力        较弱 (通信失败通常 abort)         较好 (支持超时和错误恢复)
```

### 7.3 MPI vs NCCL/Gloo

```
MPI (Message Passing Interface):
  - 是一个标准/规范，不是具体实现
  - 实现: OpenMPI, MPICH, Intel MPI, MVAPICH 等
  - 历史最悠久，原语最完整 (100+ 个 API)
  - HPC 领域标配，但不是为 GPU 原生设计
  - 可通过 CUDA-aware MPI 支持 GPU，但性能不如 NCCL

为什么深度学习不直接用 MPI?
  1. MPI 进程模型重量级 (fork/exec)，不如 NCCL 轻量
  2. MPI 对 GPU 显存通信的优化不如 NCCL 深
  3. NCCL 能感知 NVLink/NVSwitch 拓扑做最优路由
  4. 但 MPI 仍用于: 启动进程 (mpirun)、复杂通信模式、HPC 混合负载
```

### 7.4 PyTorch 中的典型搭配

```python
# GPU 分布式训练 (最常见):
torch.distributed.init_process_group(backend="nccl")
# → 梯度同步 AllReduce 走 NCCL (高性能)
# → 内部 barrier/进程管理可能仍走 Gloo

# 纯 CPU 训练或推理:
torch.distributed.init_process_group(backend="gloo")

# 使用 MPI (较少见):
torch.distributed.init_process_group(backend="mpi")
# → 需要编译安装 MPI 版本的 PyTorch
```


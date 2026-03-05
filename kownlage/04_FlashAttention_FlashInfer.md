# 4. FlashAttention 和 FlashInfer 详解

---

## 4.1 背景：标准 Attention 的瓶颈

### 4.1.1 标准 Attention 的计算流程

标准的 Self-Attention 计算过程如下：

```
输入: Q, K, V ∈ R^{N×d}    （N=序列长度, d=头维度）

Step 1: S = Q × K^T          → S ∈ R^{N×N}    （计算注意力分数矩阵）
Step 2: P = softmax(S)       → P ∈ R^{N×N}    （对每行做 softmax）
Step 3: O = P × V            → O ∈ R^{N×d}    （加权求和得到输出）
```

### 4.1.2 瓶颈分析

**计算复杂度**：O(N^2 × d)，随序列长度二次增长。

**内存瓶颈（更关键）**：
- 中间矩阵 S 和 P 的大小为 N×N。当 N=4096、d=128 时，S 矩阵需要 4096×4096×2bytes(FP16) ≈ 32MB。
- 这些中间矩阵必须在 GPU 的 HBM（High Bandwidth Memory，高带宽内存）中存储。
- 问题不在于计算（GPU 算力充足），而在于**内存读写**：GPU 需要反复在 HBM 和计算单元之间搬运数据。

**GPU 内存层次结构**：
```
┌────────────────────────────────────┐
│  SRAM（片上高速缓存）                │
│  容量: ~20MB (A100)                 │
│  带宽: ~19 TB/s                     │
│  特点: 极快但容量小                   │
├────────────────────────────────────┤
│  HBM（高带宽内存）                   │
│  容量: 40-80GB (A100)               │
│  带宽: ~2 TB/s                      │
│  特点: 容量大但相对慢                 │
└────────────────────────────────────┘
```

SRAM 带宽约为 HBM 的 10 倍，但容量小得多。标准 Attention 需要将 N×N 的中间矩阵写入 HBM 再读回，造成严重的 IO 瓶颈。

---

## 4.2 FlashAttention v1：核心突破

**论文**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)

### 4.2.1 三大核心思想

#### （1）IO-Aware（IO 感知）

FlashAttention 的核心洞察是：**Attention 的瓶颈不是计算量（FLOPs），而是内存读写（IO）**。

传统优化关注减少 FLOPs，但 FlashAttention 关注的是减少 HBM 的访问次数。这就是"IO-Aware"的含义——算法设计时明确考虑 GPU 的内存层次结构，优化数据在 SRAM 和 HBM 之间的搬运。

**标准 Attention 的 HBM 访问模式**：
```
HBM 读取 Q, K → 计算 S = QK^T → 写入 S 到 HBM
HBM 读取 S → 计算 P = softmax(S) → 写入 P 到 HBM
HBM 读取 P, V → 计算 O = PV → 写入 O 到 HBM
```
总共进行了 **3 次读 + 3 次写** 大矩阵到 HBM，这就是瓶颈所在。

#### （2）Tiling（分块计算）

核心思路：**将 Q、K、V 分成小块（tiles），每块可以放入 SRAM 中，在 SRAM 中完成尽可能多的计算，避免将中间结果写回 HBM**。

```
将 Q 分成 T_r 个块: Q_1, Q_2, ..., Q_{T_r}，每块大小 B_r × d
将 K, V 分成 T_c 个块: K_1, ..., K_{T_c} 和 V_1, ..., V_{T_c}，每块大小 B_c × d

对于每个 Q 块 Q_i：
    初始化: O_i = 0, l_i = 0 (行求和), m_i = -∞ (行最大值)
    对于每个 K, V 块 (K_j, V_j)：
        1. 从 HBM 加载 Q_i, K_j, V_j 到 SRAM
        2. 在 SRAM 中计算 S_{ij} = Q_i × K_j^T
        3. 在 SRAM 中计算局部 softmax（使用 online softmax）
        4. 更新 O_i（累积加权求和）
    将最终的 O_i 写回 HBM
```

**关键优势**：整个 N×N 的注意力矩阵 S 永远不需要被完整地存储在 HBM 中！每次只在 SRAM 中处理一小块。

#### （3）Online Softmax（在线 Softmax）

这是 FlashAttention 能做分块计算的数学基础。标准 softmax 需要看到一行的所有值才能计算（因为需要全局最大值和求和），但 Online Softmax 可以**逐块增量更新**。

**标准 Softmax**：
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

为了数值稳定性，实际计算：
m = max(x)
softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)
```

这需要对整行做两遍扫描：第一遍找最大值 m，第二遍计算 exp 和求和。

**Online Softmax 的增量更新**：

假设我们已经处理了前 j-1 个 K 块，得到了：
- m^{(j-1)}: 已见到的最大值
- l^{(j-1)}: 已见到的 exp 求和
- O^{(j-1)}: 已累积的加权输出

现在处理第 j 个 K 块：
```
# 计算当前块的注意力分数
S_j = Q_i × K_j^T

# 更新最大值
m_new^{(j)} = max(m^{(j-1)}, rowmax(S_j))

# 更新 exp 求和 (关键：需要用新的最大值重新缩放旧的求和)
l^{(j)} = exp(m^{(j-1)} - m_new^{(j)}) × l^{(j-1)} + rowsum(exp(S_j - m_new^{(j)}))

# 更新输出 (关键：旧输出也需要用新的最大值重新缩放)
O^{(j)} = exp(m^{(j-1)} - m_new^{(j)}) × O^{(j-1)} + exp(S_j - m_new^{(j)}) × V_j

# 最终归一化
O_final = O^{(T_c)} / l^{(T_c)}
```

这个更新公式确保：**无论分成多少块，最终结果和标准 Attention 完全一致**（精确计算，非近似）。

### 4.2.2 IO 复杂度分析

**标准 Attention 的 HBM 访问量**：O(N^2 × d + N^2) = O(N^2)

**FlashAttention 的 HBM 访问量**：O(N^2 × d^2 / M)，其中 M 是 SRAM 大小。

当 M >> d^2 时（通常成立，因为 d=128，d^2=16K，而 SRAM 有几十 MB），FlashAttention 的 IO 复杂度远小于标准 Attention。论文证明这个 IO 复杂度在给定 SRAM 大小范围内是**最优的**。

### 4.2.3 FlashAttention v1 的性能

| 基准测试 | 加速比 |
|---------|--------|
| BERT-large (seq=512) | 端到端 15% 加速 |
| GPT-2 (seq=1K) | 3× 加速 |
| 长序列任务 (seq=1K-4K) | 2.4× 加速 |

更重要的是，FlashAttention 首次使 Transformer 能处理 16K-64K 的超长序列。

---

## 4.3 FlashAttention v2：深度硬件优化

**论文**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)

### 4.3.1 v1 的不足

FlashAttention v1 虽然大幅减少了 HBM 访问，但在计算效率上只达到了 A100 理论峰值 FLOPs 的 **25-40%**。原因是 GPU 的**线程块（thread block）和 warp 之间的工作分配不够优化**。

### 4.3.2 三个核心改进

#### （1）减少非矩阵乘法操作

GPU 的 Tensor Core 专为矩阵乘法（GEMM）优化，throughput 远高于其他操作。v2 重新设计了算法，将更多计算转化为 GEMM 操作，减少了 softmax 中的标量运算、比较操作等非 matmul FLOPs。

#### （2）改进跨线程块的并行性

v1 中，对于单个注意力头，只在 batch 和 head 两个维度上并行。v2 额外在**序列长度维度**上并行化：

```
v1 的并行策略：
  外层循环: K, V 块（串行）
  内层循环: Q 块（并行化到不同线程块）

v2 的并行策略：
  外层循环: Q 块（并行化到不同线程块）
  内层循环: K, V 块（串行）
```

这个看似简单的循环交换带来了重大改进：
- Q 块的数量通常更多，提供了更高的并行度
- 每个线程块独立处理自己的 Q 块，无需在块间同步
- 对于因果注意力（causal attention），可以跳过不需要的 K 块，减少约 50% 的计算

#### （3）优化 warp 内工作分配

在一个线程块内部，v1 在 warp 之间分割 K 和 V，导致需要通过共享内存（shared memory）通信来汇总结果。v2 改为在 warp 之间分割 Q，每个 warp 独立处理自己的 Q 子块，**消除了 warp 间的通信开销**。

```
v1 warp 分配:
  所有 warp 共同处理同一个 Q 块
  → 需要 warp 间同步来合并结果（通过 shared memory）

v2 warp 分配:
  每个 warp 处理 Q 块的不同子块
  → 各自独立计算，无需通信
```

### 4.3.3 v2 性能

- 在 A100 上达到理论峰值的 **50-73%** FLOPs/s（v1 是 25-40%）
- 比 v1 快约 **2×**
- GPT 风格训练达到每 A100 **225 TFLOPs/s**（72% 模型 FLOPs 利用率）
- 接近优化 GEMM 操作的效率

---

## 4.4 FlashAttention v3：Hopper 架构专项优化

**论文**: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (Shah et al., 2024)

### 4.4.1 为什么需要 v3

FlashAttention v2 在 H100（Hopper 架构）上只达到 **35%** 利用率。这是因为 H100 引入了新的硬件特性（如 TMA、异步 Tensor Core 等），v2 的设计无法充分利用这些特性。

### 4.4.2 三大创新

#### （1）Warp 专门化（Warp Specialization）

利用 H100 的 TMA（Tensor Memory Accelerator）和异步 Tensor Core，将不同的 warp 专门分工：
- **Producer warp**: 负责数据加载（通过 TMA 异步从 HBM 加载到 SRAM）
- **Consumer warp**: 负责计算（使用 Tensor Core 做 GEMM）

```
传统方式（同步）：
  加载数据 → 等待 → 计算 → 等待 → 加载下一批 → ...

v3 方式（异步流水线）：
  Producer:  加载块1 | 加载块2 | 加载块3 | ...
  Consumer:         | 计算块1 | 计算块2 | ...
  （加载和计算重叠执行）
```

#### （2）交错式 matmul-softmax 流水线

在 v2 中，matmul 和 softmax 是串行执行的：先算完 S=QK^T，再做 softmax，再算 PV。v3 将这两个操作**交错执行**：

```
v2 执行序列：
  GEMM(QK^T) → softmax → GEMM(PV) → GEMM(QK^T) → softmax → GEMM(PV)

v3 执行序列（交错）：
  GEMM_1(QK^T) → [softmax_1 与 GEMM_2(QK^T) 同时执行] → [GEMM_1(PV) 与 softmax_2 同时执行] → ...
```

softmax 是非 GEMM 操作，在 Tensor Core 空闲时用普通 CUDA Core 执行，实现了计算资源的充分利用。

#### （3）FP8 低精度支持

利用 H100 原生的 FP8 硬件支持，引入两个技术保证精度：

- **块量化（Block Quantization）**：不是全局统一量化，而是对每个小块独立计算量化参数
- **Incoherent Processing（非相干处理）**：通过随机正交变换使量化误差分布更均匀，减少系统性误差

FP8 精度改进效果：比基线 FP8 attention 的**数值误差降低 2.6 倍**。

### 4.4.3 v3 性能

| 精度 | 吞吐量 | 利用率 |
|------|--------|-------|
| FP16/BF16 | 最高 740 TFLOPs/s | 75%（H100） |
| FP8 | ~1.2 PFLOPs/s | - |

- 比 FlashAttention v2 快 **1.5-2.0×**（FP16）
- H100 利用率从 35% 提升到 **75%**

---

## 4.5 三个版本总结对比

| 特性 | FlashAttention v1 | FlashAttention v2 | FlashAttention v3 |
|------|-------------------|-------------------|-------------------|
| 发布年份 | 2022 | 2023 | 2024 |
| 目标 GPU | 通用（A100 为主） | A100 优化 | H100 (Hopper) 专项优化 |
| 核心创新 | Tiling + Online Softmax + IO-Aware | 更好的并行性 + Warp 分配 | Warp 专门化 + 异步流水线 + FP8 |
| GPU 利用率 | 25-40% | 50-73% | ~75% |
| 精度 | FP16/BF16 | FP16/BF16 | FP16/BF16 + FP8 |
| 是否精确 | 是 | 是 | 是（FP8 有量化误差但已最小化） |
| 相对上代加速 | - | ~2× | 1.5-2.0× |

---

## 4.6 FlashInfer：面向推理的专业 Attention 库

### 4.6.1 什么是 FlashInfer

FlashInfer 是一个**专门面向 LLM 推理场景**的 GPU 内核库和内核生成器。虽然 FlashAttention 最初主要为训练场景设计，FlashInfer 则针对推理场景的特殊需求进行了深度优化。

### 4.6.2 核心功能

#### （1）Attention 内核

| 功能 | 说明 |
|------|------|
| **Paged KV-Cache** | 支持类似 vLLM 的 PagedAttention 内存管理 |
| **Ragged KV-Cache** | 支持变长序列的 KV-Cache 紧凑存储 |
| **Decode Attention** | 推理解码阶段的专用内核 |
| **Prefill Attention** | 推理预填充阶段的专用内核 |
| **Append Attention** | 增量追加 KV 的 Attention 内核 |
| **MLA Attention** | 原生支持 DeepSeek 的 Multi-head Latent Attention |
| **Cascade Attention** | 层级 KV-Cache，用于共享前缀场景 |
| **Sparse Attention** | 块稀疏和可变块稀疏模式支持 |
| **POD-Attention** | 融合 prefill + decode 的混合批次内核 |

#### （2）GEMM 与量化

- **BF16 GEMM**：支持 SM 10.0+（Blackwell 及更新架构）
- **FP8 GEMM**：支持 per-tensor 和 groupwise 缩放
- **FP4 GEMM**：支持 NVFP4 和 MXFP4 格式（Blackwell GPU）
- **Grouped GEMM**：用于 LoRA 适配器和多专家路由

#### （3）MoE（混合专家）

- 融合 MoE 内核，支持多种路由方式：DeepSeek-V3、Llama-4 和标准 Top-K
- 支持 FP8 和 FP4 量化的专家权重

#### （4）采样

- **Sorting-Free Sampling**：无需排序的 Top-K、Top-P、Min-P 采样
- **Chain Speculative Sampling**：用于投机解码的链式采样

#### （5）融合算子

- RoPE（包括 LLaMA 3.1 风格）
- RMSNorm、LayerNorm
- SiLU、GELU（带融合门控）
- 自定义 AllReduce、多节点 NVLink、NVSHMEM 集成

### 4.6.3 FlashInfer 与 FlashAttention 的关系

FlashInfer 并非 FlashAttention 的竞品，而是**在推理场景下对 FlashAttention 的封装和扩展**：

```
FlashInfer 的后端包括：
├── FlashAttention-2/3 内核（用于 Prefill）
├── cuDNN 内核
├── CUTLASS 内核
├── TensorRT-LLM 内核
└── 自研的 Decode 内核

FlashInfer 在此基础上增加了：
├── PagedAttention 内存管理支持
├── 动态批次的高效调度
├── 多种 Attention 变体（MLA, GQA, MQA 等）
├── JIT 编译（运行时生成最优内核）
└── CUDAGraph 兼容性
```

### 4.6.4 在推理框架中的集成

FlashInfer 已被所有主流推理框架集成：

| 框架 | 角色 |
|------|------|
| **SGLang** | 默认 Attention 后端 |
| **vLLM** | Attention 后端之一 |
| **TensorRT-LLM** | 内核后端 |
| **TGI** (Hugging Face) | Attention 后端 |
| **MLC-LLM** | 内核后端 |
| **LightLLM** | 内核后端 |

**集成方式示意**：
```python
# 以 SGLang 为例，FlashInfer 的使用方式
import flashinfer

# 1. 创建 Paged KV-Cache
workspace = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,    # 工作空间
    kv_layout="NHD",     # KV 缓存布局
)

# 2. 在 Decode 阶段调用
workspace.plan(
    indptr,              # 每个请求的页表偏移
    indices,             # 页索引
    last_page_len,       # 最后一页的有效长度
    num_qo_heads,        # Query/Output 头数
    num_kv_heads,        # KV 头数（用于 GQA/MQA）
    head_dim,            # 头维度
    page_size,           # 页大小
)

output = workspace.run(q, paged_kv_cache)
```

### 4.6.5 性能特性

- **CUDAGraph 兼容**：支持 CUDAGraph 捕获，消除内核启动开销，实现低延迟推理
- **torch.compile 兼容**：可与 PyTorch 编译器集成
- **JIT 编译**：运行时根据具体场景生成最优内核
- **预编译内核缓存**：flashinfer-cubin 和 flashinfer-jit-cache 包可加速启动
- **广泛的 GPU 架构支持**：从 Turing (SM 7.5) 到 Blackwell (SM 10.0+)

### 4.6.6 FlashInfer 解决的推理特有问题

推理场景与训练场景有本质区别：

| 特性 | 训练 | 推理 |
|------|------|------|
| 序列长度 | 固定 | 动态变化 |
| 批次大小 | 固定 | 动态变化 |
| 计算模式 | 全序列 Attention | Prefill + Decode 两阶段 |
| KV Cache | 不需要 | 核心需求 |
| 内存管理 | 简单（预分配） | 复杂（动态分配/释放） |

FlashInfer 专门针对这些推理特有需求做了深度优化，而不是简单复用训练时的 FlashAttention 内核。

---

## 4.7 与标准 Attention 的全面性能对比

### 4.7.1 内存使用

```
标准 Attention:  O(N^2)  — 需要存储完整的 N×N 注意力矩阵
FlashAttention: O(N)    — 只需存储 O(N) 的输出和少量统计量

示例 (N=8192, d=128, FP16):
  标准 Attention: N×N×2 = 128 MB（中间矩阵）
  FlashAttention: ~几 MB（分块处理，不存储完整矩阵）
```

### 4.7.2 速度对比

```
在 A100 GPU 上（序列长度 2048，头数 16，头维度 128）：
  标准 Attention (PyTorch):  ~100 TFLOPs/s
  FlashAttention v1:          ~130 TFLOPs/s  (1.3×)
  FlashAttention v2:          ~230 TFLOPs/s  (2.3×)

在 H100 GPU 上：
  FlashAttention v2:          ~350 TFLOPs/s
  FlashAttention v3 (FP16):   ~740 TFLOPs/s  (2.1×)
  FlashAttention v3 (FP8):    ~1200 TFLOPs/s (3.4×)
```

### 4.7.3 序列长度扩展能力

```
标准 Attention 最大可处理序列长度（受 GPU 内存限制）:
  A100 40GB: ~16K
  A100 80GB: ~32K

FlashAttention 最大可处理序列长度:
  A100 40GB: ~64K+
  A100 80GB: ~128K+
```

FlashAttention 使得超长上下文窗口（如 128K、1M）成为可能，这是现代大模型（GPT-4、Claude、Gemini）能支持长上下文的关键技术基础。

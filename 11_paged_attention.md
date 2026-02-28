# 11. Paged Attention 详解

## 一、核心思想：借鉴操作系统虚拟内存分页机制管理 KV Cache

### 1.1 问题背景：为什么需要 Paged Attention

在大模型推理中，**KV Cache** 是最关键的显存消耗来源。让我们先理解问题：

#### 传统 KV Cache 管理的困境

```
场景：一个 13B 模型，max_seq_len=2048，batch_size=8

单个请求的 KV Cache 显存 = 2 * num_layers * num_heads * head_dim * seq_len * dtype_size
= 2 * 40 * 40 * 128 * 2048 * 2 bytes (FP16)
≈ 1.6 GB

8 个并发请求 = 8 * 1.6 GB ≈ 12.8 GB
```

传统做法是**预分配连续显存**：
```
┌─────────────────────────────────────────────────┐
│ GPU 显存                                         │
│                                                  │
│ [Request 1 KV Cache: 预分配 2048 tokens 空间]    │  ← 实际只用了 500 tokens
│ [Request 2 KV Cache: 预分配 2048 tokens 空间]    │  ← 实际只用了 200 tokens
│ [Request 3 KV Cache: 预分配 2048 tokens 空间]    │  ← 实际只用了 1500 tokens
│ [        空闲空间（碎片化）                  ]    │
│                                                  │
└─────────────────────────────────────────────────┘
```

**三大问题：**

1. **显存浪费（Internal Fragmentation）**
   - 必须按 max_seq_len 预分配，但大多数请求不会用满
   - 平均利用率可能只有 30-50%
   - 一个请求只生成了 100 个 token，但预分配了 2048 个 token 的空间

2. **显存碎片化（External Fragmentation）**
   - 请求完成后释放的空间大小不一
   - 新请求可能找不到足够大的连续空间
   - 即使总空闲显存足够，也无法分配

3. **无法动态增长**
   - 预分配模式下，序列长度无法动态增长
   - 如果预分配太小，长序列无法处理
   - 如果预分配太大，短序列浪费严重

### 1.2 操作系统虚拟内存的启发

操作系统的虚拟内存系统完美解决了类似问题：

```
操作系统虚拟内存：
┌─────────────┐        ┌─────────────────────────┐
│ 虚拟地址空间  │  页表   │ 物理内存                  │
│             │  ───>  │                          │
│ Page 0 ─────┼────────┼──> Physical Frame 5     │
│ Page 1 ─────┼────────┼──> Physical Frame 2     │
│ Page 2 ─────┼────────┼──> Physical Frame 8     │
│ Page 3      │ (未映射) │                          │
│ ...         │        │                          │
└─────────────┘        └─────────────────────────┘

特点：
- 虚拟地址连续，物理地址可以不连续
- 按需分配物理页面（page fault 时才分配）
- 固定大小的 page 消除外部碎片
- 最多浪费一个 page 的空间（最后一个 page 可能未填满）
```

**Paged Attention 的核心类比：**

| 操作系统概念 | Paged Attention 对应概念 |
|-------------|------------------------|
| 虚拟页面（Virtual Page）| 逻辑 KV Block |
| 物理页帧（Physical Frame）| 物理 KV Block（GPU 显存中的实际块）|
| 页表（Page Table）| Block Table（块映射表）|
| 页面大小（Page Size）| Block Size（通常 16 tokens）|
| 按需分配（Demand Paging）| 生成新 token 时才分配新 block |
| 页面回收 | 请求完成后回收 block |

---

## 二、实现原理

### 2.1 基本数据结构

#### Block（物理块）

```python
# 每个 Block 存储固定数量 token 的 KV Cache
class KVBlock:
    block_size: int = 16  # 每个 block 存 16 个 token 的 KV
    # 实际存储的张量形状：
    # Key:   [block_size, num_heads, head_dim]
    # Value: [block_size, num_heads, head_dim]

    # 物理存储在 GPU 显存的一个连续区域中
    # key_cache:   GPU tensor of shape [num_blocks, block_size, num_kv_heads, head_dim]
    # value_cache: GPU tensor of shape [num_blocks, block_size, num_kv_heads, head_dim]
```

#### Block Table（块映射表）

```python
# Block Table 记录每个序列的逻辑块到物理块的映射
# 类似操作系统的页表

# 例如一个序列有 48 个 token，block_size=16
# 逻辑块: [0, 1, 2]  → 物理块: [7, 3, 15]

block_table = {
    seq_id_0: [7, 3, 15],      # 3 个 block
    seq_id_1: [1, 9],           # 2 个 block
    seq_id_2: [4, 12, 6, 20],  # 4 个 block
}
```

#### Free Block Pool（空闲块池）

```python
# 全局维护一个空闲物理块的池子
free_blocks = [0, 2, 5, 8, 10, 11, 13, 14, 16, 17, 18, 19, ...]

# 分配：从池子中取出一个块
def allocate_block():
    return free_blocks.pop()

# 回收：将块归还池子
def free_block(block_id):
    free_blocks.append(block_id)
```

### 2.2 整体内存布局

```
GPU 显存中的 KV Cache 布局（物理视图）：

┌────────────────────────────────────────────────────────────────┐
│ KV Cache Pool（预分配的一大块连续显存，被划分为等大的 Block）     │
│                                                                │
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │...│
│ (free)  │ (seq 1) │ (free)  │ (seq 0) │ (seq 2) │ (free)  │   │
│         │ blk #1  │         │ blk #1  │ blk #0  │         │   │
└────────────────────────────────────────────────────────────────┘

Block Table（逻辑视图）：
Sequence 0: 逻辑块 [0] → 物理块 [3]        (16 tokens)
Sequence 1: 逻辑块 [0, 1] → 物理块 [7, 1]   (32 tokens)
Sequence 2: 逻辑块 [0] → 物理块 [4]        (10 tokens，最后一个块部分填充)
```

### 2.3 Paged Attention 的计算过程

传统 Attention 计算：
```python
# 标准 Attention
# Q: [batch, num_heads, 1, head_dim]（当前 token）
# K: [batch, num_heads, seq_len, head_dim]（所有历史 token）
# V: [batch, num_heads, seq_len, head_dim]
output = softmax(Q @ K^T / sqrt(d)) @ V
```

Paged Attention 的计算：
```python
# Paged Attention 需要根据 Block Table 从不连续的物理块中收集 KV
# 核心挑战：K 和 V 不再是连续的，而是分散在不同的物理块中

def paged_attention(query, key_cache, value_cache, block_table, context_len):
    """
    query:      [num_heads, head_dim] - 当前 token 的 query
    key_cache:  [num_blocks, block_size, num_kv_heads, head_dim] - 全局 key 缓存
    value_cache:[num_blocks, block_size, num_kv_heads, head_dim] - 全局 value 缓存
    block_table:[max_num_blocks_per_seq] - 该序列的物理块映射
    context_len: int - 该序列当前的长度
    """
    output = zeros(num_heads, head_dim)

    num_blocks = ceil(context_len / block_size)

    for block_idx in range(num_blocks):
        physical_block = block_table[block_idx]

        # 从物理块中取出 K 和 V
        if block_idx == num_blocks - 1:
            # 最后一个块可能没填满
            tokens_in_block = context_len - block_idx * block_size
        else:
            tokens_in_block = block_size

        k_block = key_cache[physical_block, :tokens_in_block]    # [tokens, heads, dim]
        v_block = value_cache[physical_block, :tokens_in_block]  # [tokens, heads, dim]

        # 计算该块的 attention
        attn_scores = query @ k_block.T / sqrt(head_dim)  # [heads, tokens]
        # 累积（需要做 online softmax，类似 FlashAttention 的思路）
        output = update_output(output, attn_scores, v_block)

    return output
```

### 2.4 Paged Attention CUDA Kernel 实现要点

vLLM 的 Paged Attention kernel 是一个高度优化的 CUDA kernel：

```
Kernel 设计要点：

1. 每个 Thread Block 处理一个 attention head
2. 每个 Warp 处理一个或多个 KV Block

并行策略：
- Grid 维度: (num_seqs, num_heads, num_partitions)
- Block 维度: (BLOCK_SIZE,) 或 (NUM_THREADS,)

计算流程：
For each KV block:
    1. 根据 block_table 计算物理块地址
    2. 从 key_cache 加载 K block 到 shared memory
    3. 计算 Q·K^T 得到 attention scores
    4. Online softmax 更新全局 max 和 sum
    5. 从 value_cache 加载 V block
    6. 用 attention weights 加权求和 V
    7. 更新 output accumulator
```

**两个版本的 Paged Attention Kernel：**

- **V1**：单个 sequence 的所有 KV blocks 由一个 thread block 处理
  - 适用于较短的序列
  - 每个 thread block 顺序遍历所有 KV blocks

- **V2**：将一个 sequence 的 KV blocks 分给多个 thread block 并行处理
  - 适用于长序列
  - 需要一个额外的 reduction step 合并结果
  - 类似 FlashAttention 的分块并行思路

---

## 三、解决的问题

### 3.1 解决显存碎片化

**Before（传统连续分配）：**
```
GPU Memory:
[Seq A: 2048 slots][   free   ][Seq B: 2048 slots][free][Seq C: 2048 slots]
                   ↑ 碎片                        ↑ 碎片
                   (太小，放不下新请求)             (太小)

总空闲 = 1000 slots，但分成了两个小块，新请求需要 2048 连续 slots → 分配失败！
```

**After（Paged Attention）：**
```
GPU Memory (Block Pool):
[B0:A][B1:A][B2:free][B3:B][B4:free][B5:C][B6:free][B7:A][B8:free]...

所有 free blocks 组成一个池子：{B2, B4, B6, B8, ...}
新请求只需要从池子中取出足够数量的 blocks，不要求连续！
→ 分配成功！

外部碎片完全消除！
```

### 3.2 解决显存浪费

**Before：**
```
Request 1: 预分配 2048 tokens 空间，实际生成了 150 tokens → 浪费率 92.7%
Request 2: 预分配 2048 tokens 空间，实际生成了 500 tokens → 浪费率 75.6%
Request 3: 预分配 2048 tokens 空间，实际生成了 2000 tokens → 浪费率 2.3%
平均浪费率约 57%
```

**After（Paged Attention, block_size=16）：**
```
Request 1: 150 tokens → 需要 ceil(150/16) = 10 个 blocks → 浪费 10 tokens → 浪费率 6.3%
Request 2: 500 tokens → 需要 ceil(500/16) = 32 个 blocks → 浪费 12 tokens → 浪费率 2.4%
Request 3: 2000 tokens → 需要 ceil(2000/16) = 125 个 blocks → 浪费 0 tokens → 浪费率 0%
平均浪费率 < 4%（最多浪费一个 block）
```

**vLLM 论文的关键数据：Paged Attention 将显存浪费从 60-80% 降低到不到 4%。**

### 3.3 支持动态序列增长

```
Time Step 1: Sequence 有 15 tokens，占用 1 个 block [B3]
Time Step 2: 生成第 16 个 token，block [B3] 刚好填满
Time Step 3: 生成第 17 个 token → 从 free pool 分配新 block [B7]
             Block table: [B3, B7]
...
Time Step N: 序列结束，回收所有 blocks: B3, B7 → free pool
```

不需要预分配，不需要重新分配和复制，完全按需增长。

### 3.4 支持更大的 Batch Size

由于显存利用率从 ~40% 提升到 ~96%，**同样的 GPU 显存可以支持 2-4 倍的并发请求数**。

```
传统方式：80GB GPU，每个请求预分配 1.6GB → 最多 ~35 并发
Paged Attention：80GB GPU，平均每个请求实际使用 0.4GB → 最多 ~150 并发
                  （假设平均序列长度只有 max_seq_len 的 25%）
```

更大的 batch size 意味着更高的 GPU 利用率和更高的吞吐量。

---

## 四、vLLM 中的具体实现

### 4.1 核心类和数据结构

```python
# vLLM 的关键类（简化版）

class BlockAllocator:
    """管理物理块的分配和回收"""
    def __init__(self, num_blocks, block_size):
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        """分配一个物理块"""
        return self.free_blocks.pop()

    def free(self, block_id: int):
        """回收一个物理块"""
        self.free_blocks.append(block_id)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class BlockSpaceManager:
    """管理所有序列的 block 分配"""
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = block_size
        self.gpu_allocator = BlockAllocator(num_gpu_blocks, block_size)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks, block_size)  # 用于 swap
        self.block_tables: Dict[int, List[int]] = {}  # seq_id → [physical_block_ids]

    def allocate(self, seq_id: int):
        """为新序列分配初始 blocks"""
        num_blocks_needed = ceil(seq_len / self.block_size)
        blocks = [self.gpu_allocator.allocate() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = blocks

    def append_slot(self, seq_id: int):
        """为序列追加一个 token 的空间"""
        block_table = self.block_tables[seq_id]
        last_block = block_table[-1]

        if self._is_block_full(last_block):
            # 最后一个 block 已满，分配新 block
            new_block = self.gpu_allocator.allocate()
            block_table.append(new_block)

    def swap_out(self, seq_id: int):
        """将序列的 KV Cache 从 GPU 换出到 CPU"""
        gpu_blocks = self.block_tables[seq_id]
        cpu_blocks = [self.cpu_allocator.allocate() for _ in gpu_blocks]
        # 异步 GPU→CPU 数据传输
        copy_blocks(gpu_blocks, cpu_blocks, gpu_to_cpu=True)
        # 更新映射
        for gb in gpu_blocks:
            self.gpu_allocator.free(gb)
        self.block_tables[seq_id] = cpu_blocks  # 标记为在 CPU 上

    def swap_in(self, seq_id: int):
        """将序列的 KV Cache 从 CPU 换回 GPU"""
        # 类似 swap_out 的反向操作
        ...
```

### 4.2 vLLM 的调度器如何与 Paged Attention 配合

```python
class Scheduler:
    def schedule(self):
        """每一步推理的调度决策"""

        # 1. 检查是否有序列需要被抢占（preempt）
        #    当 GPU blocks 不够时，将低优先级序列 swap out
        while self.block_manager.get_num_free_blocks() < needed:
            victim = self.running_queue.get_lowest_priority()
            self.block_manager.swap_out(victim.seq_id)
            self.swapped_queue.add(victim)
            self.running_queue.remove(victim)

        # 2. 为 running 中的序列分配新 token 的 block
        for seq in self.running_queue:
            self.block_manager.append_slot(seq.seq_id)

        # 3. 尝试 swap in 之前被换出的序列
        while self.swapped_queue and self.block_manager.has_enough_blocks():
            seq = self.swapped_queue.pop()
            self.block_manager.swap_in(seq.seq_id)
            self.running_queue.add(seq)

        # 4. 尝试接纳新请求
        while self.waiting_queue and self.block_manager.has_enough_blocks():
            seq = self.waiting_queue.pop()
            self.block_manager.allocate(seq.seq_id)
            self.running_queue.add(seq)

        # 返回调度决策
        return SchedulerOutput(
            running_seqs=self.running_queue,
            block_tables=self.block_manager.get_block_tables(),
            ...
        )
```

### 4.3 Copy-on-Write（写时复制）

Paged Attention 还支持类似操作系统的 **Copy-on-Write** 机制，用于 **Parallel Sampling**（如 beam search）：

```
场景：beam search 中，一个序列分裂成多个候选

原始序列: "The cat sat on the"
Block Table: [B3, B7, B12]

分裂为 3 个 beam：
Beam 1: "The cat sat on the mat"
Beam 2: "The cat sat on the floor"
Beam 3: "The cat sat on the chair"

Without CoW（Copy-on-Write）:
  需要复制 3 份完整的 KV Cache → 3 × [B3, B7, B12] = 9 个 blocks

With CoW:
  共享前缀的 blocks，只在修改时复制
  Beam 1: [B3, B7, B12, B20]     ← B20 是新分配的
  Beam 2: [B3, B7, B12, B21]     ← B3, B7, B12 共享，引用计数 = 3
  Beam 3: [B3, B7, B12, B22]     ← 只用了 6 个 blocks（而非 12 个）

  当某个 beam 需要修改共享 block 时，才执行复制
  引用计数管理：每个 block 维护引用计数，归零时回收
```

---

## 五、与 FlashInfer PagedAttention 的关系

### 5.1 FlashInfer 简介

FlashInfer 是一个专门为 LLM serving 场景优化的 **attention kernel 库**，由 CMU 的研究者开发。它提供了高度优化的 Paged Attention CUDA kernel 实现。

### 5.2 vLLM 原生 Paged Attention vs FlashInfer Paged Attention

```
vLLM 原始架构：
  Scheduler (Python) → vLLM Paged Attention Kernel (CUDA C++) → GPU

SGLang / 新版 vLLM：
  Scheduler (Python) → FlashInfer Paged Attention Kernel (CUDA C++) → GPU
```

**vLLM 原生 Paged Attention Kernel：**
- vLLM 项目自己编写的 CUDA kernel
- 包含 V1 和 V2 两个版本
- V1：一个 thread block 处理一个 head 的所有 KV blocks
- V2：多个 thread block 并行处理一个 head（适合长序列），需要额外的 reduce

**FlashInfer Paged Attention Kernel：**
- 独立的开源库，专注于 attention kernel 优化
- 基于 FlashAttention 的思路，结合 Paged KV Cache
- 支持更多优化特性

### 5.3 FlashInfer 的优势

1. **更好的 Prefill 性能**
   - FlashInfer 在 prefill 阶段使用了类似 FlashAttention 的分块策略
   - 对长 prompt 的 prefill 性能更好

2. **Ragged Tensor 支持**
   - 支持不同长度序列的高效 batched attention
   - 避免了 padding 造成的计算浪费

3. **更灵活的 Block 大小**
   - 支持多种 block size（1, 4, 8, 16, 32 等）
   - vLLM 原生 kernel 通常固定为 16

4. **Cascade Attention / Multi-level Paging**
   - 支持多级页表，进一步优化 prefix sharing 场景
   - 共享前缀可以用大块（如 128 tokens），独有部分用小块（如 16 tokens）

5. **与 CUDAGraph 更好的兼容性**
   - FlashInfer 的 API 设计更适合与 CUDA Graph 配合
   - 通过 plan/run 两步 API，把动态部分放在 plan（CPU），静态部分放在 run（GPU capture）

### 5.4 SGLang 与 FlashInfer 的紧密关系

SGLang 从一开始就深度集成 FlashInfer：
- SGLang 的默认 attention backend 是 FlashInfer
- FlashInfer 的 RadixAttention 支持与 SGLang 的 RadixAttention 配合
- 两者的作者有密切合作

### 5.5 当前生态现状

```
vLLM:
  - 同时支持 vLLM 原生 kernel 和 FlashInfer backend
  - 通过 --attention-backend 参数切换
  - 新版本逐渐向 FlashInfer 靠拢

SGLang:
  - 默认且推荐使用 FlashInfer
  - FlashInfer 的 Paged Attention 是 SGLang 的核心依赖

TensorRT-LLM:
  - 有自己的 Paged Attention 实现
  - 不使用 FlashInfer

其他框架:
  - FlashInfer 正在成为事实上的 Paged Attention 标准库
  - 类似 FlashAttention 成为标准 attention 库的趋势
```

---

## 六、Block 大小选择对性能的影响

### 6.1 Block Size 的权衡

Block Size（每个 block 存储的 token 数）是一个关键的超参数：

| Block Size | 优点 | 缺点 |
|-----------|------|------|
| 小（1-4）| 显存浪费极少（最多浪费几个 token）| Block Table 大、管理开销高、kernel 效率低 |
| 中（8-16）| 平衡显存利用率和 kernel 效率 | 折中方案 |
| 大（32-128）| Kernel 效率高（更好的内存访问模式）| 可能浪费较多显存（最后一个块）|

### 6.2 对 Kernel 性能的影响

```
Block Size = 1（token-level）:
  - 每个 block 只有一个 token
  - Block table 非常大：seq_len=4096 → 4096 个 block entries
  - Kernel 需要做 4096 次间接寻址
  - 内存访问模式差（不连续，无法利用 memory coalescing）
  - 但显存浪费为 0

Block Size = 16（典型选择）:
  - seq_len=4096 → 256 个 block entries
  - Kernel 内部可以连续访问 16 个 token
  - 良好的 memory coalescing
  - 平均浪费 8 个 token 的空间
  - 这是 vLLM 的默认选择

Block Size = 128:
  - seq_len=4096 → 32 个 block entries
  - Block table 很小，间接寻址开销最小
  - 每个 block 内部完全连续，memory access 最优
  - 但平均浪费 64 个 token 的空间
  - 对短序列尤其不友好
```

### 6.3 不同场景的推荐

```
短序列高并发（如 chatbot，平均 100 tokens）:
  → Block Size = 8 或 16，减少浪费

长序列（如 RAG/文档处理，数千 tokens）:
  → Block Size = 16 或 32，提高 kernel 效率

Prefix Sharing 场景:
  → FlashInfer 的多级 paging：共享前缀用大块，独有部分用小块

Beam Search:
  → 较小的 Block Size 更适合 CoW（减少复制开销）
```

### 6.4 实际性能数据参考

根据 vLLM 和 FlashInfer 的 benchmark：
- Block Size 从 1 到 16，kernel 性能可以提升 **2-3x**
- Block Size 从 16 到 32，kernel 性能提升约 **10-20%**
- Block Size 超过 32，性能提升趋于饱和
- Block Size = 16 是目前最广泛使用的默认值

---

## 七、面试要点总结

1. **核心类比**：Paged Attention 借鉴了操作系统虚拟内存分页机制——物理块（Frame）、逻辑块（Page）、块表（Page Table）、按需分配（Demand Paging）
2. **解决的三大问题**：显存浪费（内部碎片→最多浪费一个 block）、显存碎片化（外部碎片→完全消除）、动态增长
3. **vLLM 的核心贡献**：提出 Paged Attention，将显存利用率从约 40% 提升到约 96%，使得 batch size 可以增大 2-4 倍
4. **FlashInfer 的角色**：更高性能的 Paged Attention kernel 实现，支持更多优化特性（cascade attention、多级 paging、更好的 CUDAGraph 兼容性）
5. **Block Size 选择**：16 是最常用的默认值，是显存利用率和 kernel 效率的平衡点
6. **Copy-on-Write**：支持 beam search 等场景下的高效共享

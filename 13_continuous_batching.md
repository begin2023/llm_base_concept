# 13. Continuous Batching（连续批处理）

---

## 一、背景与动机

大语言模型（LLM）推理的核心瓶颈在于 GPU 利用率。在自回归生成（autoregressive generation）过程中，模型每次只生成一个 token，每个 decode step 都需要完整地执行一遍前向传播。如果只处理单个请求，GPU 的计算单元大部分时间处于空闲状态，因为单个请求的 decode 阶段是 memory-bound（内存带宽受限）的操作，无法充分利用 GPU 的计算能力。

因此，"batching"（批处理）成为提升 GPU 利用率和系统吞吐量的关键技术。

---

## 二、Static Batching（静态批处理）—— 传统方式

### 2.1 基本工作原理

静态批处理是最朴素的 batching 策略：

1. **收集阶段**：系统等待收集到足够多的请求（或者等待超时），组成一个 batch。
2. **处理阶段**：将这个 batch 中所有请求一起送入模型，所有请求同步执行 prefill 和 decode。
3. **完成阶段**：等待 batch 中所有请求都生成完毕（遇到 EOS token 或达到最大长度），才开始处理下一个 batch。

### 2.2 致命缺陷

**问题 1：长尾等待（Tail Latency）**
- 假设一个 batch 中有 8 个请求，其中 7 个只需要生成 20 个 token，但有 1 个需要生成 500 个 token。在静态批处理下，那 7 个已完成的请求必须等待第 500 个 token 生成完毕才能返回结果。
- 这导致了严重的延迟浪费。

**问题 2：GPU 利用率逐步下降**
- 随着 batch 中的请求陆续完成，实际在做有效计算的请求数量越来越少，但 GPU 仍然为已完成的请求分配了 padding，导致无效计算。
- 例如：batch=8，当 5 个完成后，GPU 只在为 3 个请求做有效计算，利用率下降到约 37.5%。

**问题 3：新请求排队等待**
- 新到达的请求必须等当前 batch 全部完成后才能被处理，导致排队延迟增大。

**示意图（Static Batching 时间线）**：
```
时间轴 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
Req A: [===prefill===][=decode 20 tokens=][----padding/等待----]
Req B: [===prefill===][=====decode 50 tokens=====][--等待---]
Req C: [===prefill===][==========decode 100 tokens==========]
         ↑ 组成batch        ↑ A完成但无法返回   ↑ batch全部完成才能处理下一个batch
```

---

## 三、Continuous Batching（连续批处理）—— 核心概念

### 3.1 定义

Continuous Batching（连续批处理），也称为 "In-flight Batching" 或 "Dynamic Batching"，是一种允许在推理过程中动态调整 batch 组成的调度策略。

**核心思想**：不需要等待整个 batch 完成，而是在每一个 iteration（每一个 decode step）都可以：
- 将已完成的请求移出 batch
- 将新到达的请求插入 batch

### 3.2 Iteration-Level Scheduling（迭代级调度）

这是 Continuous Batching 的核心机制。

- 传统的 Static Batching 是 **Request-Level Scheduling**（请求级调度）：以整个请求的生命周期为调度单位。
- Continuous Batching 是 **Iteration-Level Scheduling**（迭代级调度）：以每一个 decode iteration 为调度单位。

具体含义：
- 在每一次模型前向传播（每个 decode step）之前，调度器（scheduler）都会重新评估当前 batch 的组成
- 检查是否有请求已经完成（生成了 EOS 或达到最大长度）
- 检查是否有新请求在等待队列中
- 动态调整 batch：移出完成的请求，加入新的请求
- 然后执行本次 decode step

**示意图（Continuous Batching 时间线）**：
```
时间轴 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
Req A: [=prefill=][=decode 20=]  ← 完成，立即返回
Req B: [=prefill=][====decode 50====]  ← 完成，立即返回
Req C: [=prefill=][=========decode 100=========]
Req D:            [=prefill=][==decode 30==]  ← A完成后D插入
Req E:                       [=prefill=][decode 40]  ← B完成后E插入
```

---

## 四、实现机制详解

### 4.1 核心数据结构

**（1）Waiting Queue（等待队列）**
- 存放新到达但尚未被处理的请求
- 按到达时间排序（或优先级排序）

**（2）Running Batch（运行中的 batch）**
- 当前正在 GPU 上进行推理的请求集合
- 有一个最大容量限制（受 GPU 内存和计算能力约束）

**（3）Sequence State（序列状态）**
- 每个请求维护自己的状态：
  - `WAITING`：在等待队列中
  - `RUNNING_PREFILL`：正在执行 prefill 阶段
  - `RUNNING_DECODE`：正在执行 decode 阶段
  - `FINISHED`：已完成，等待输出
  - `SWAPPED`：被换出到 CPU 内存（在内存不足时）

### 4.2 调度循环（Scheduling Loop）

每一个 iteration 的调度流程如下：

```python
# Step 1: 检查完成的请求
for request in running_batch:
    if request.last_token == EOS or request.length >= max_length:
        mark request as FINISHED
        release GPU memory (KV Cache blocks)
        remove from running_batch
        send response to client

# Step 2: 检查是否可以加入新请求
available_slots = max_batch_size - len(running_batch)
available_memory = free_gpu_memory()

while waiting_queue is not empty and available_slots > 0:
    next_request = waiting_queue.peek()
    memory_needed = estimate_memory(next_request)
    if memory_needed <= available_memory:
        request = waiting_queue.pop()
        running_batch.add(request)
        available_slots -= 1
        available_memory -= memory_needed
    else:
        break  # 内存不足，停止加入新请求

# Step 3: 执行本次 decode step
if running_batch is not empty:
    outputs = model.forward(running_batch)
    for request, output in zip(running_batch, outputs):
        request.append_token(output.token)

# Step 4: 回到 Step 1
```

### 4.3 Prefill 与 Decode 的混合处理

新加入的请求需要先执行 prefill（处理输入 prompt），而已有的请求处于 decode 阶段。这两种操作的计算特性不同：

- **Prefill**：compute-bound（计算密集），处理大量输入 token
- **Decode**：memory-bound（内存带宽密集），每次只处理一个 token

**策略 1：分离式处理（Separate Prefill and Decode）**
- 新请求的 prefill 和正在运行的 decode 分开执行
- 先为新请求执行一次完整的 prefill，然后将新请求加入 decode batch
- 缺点：prefill 期间 decode 请求需要等待

**策略 2：混合式处理（Chunked Prefill / Hybrid Batching）**
- 将 prefill 拆分成多个 chunk
- 每个 iteration 中，新请求处理一部分 prefill token，同时已有请求正常执行 decode
- 优点：不会因为 prefill 阻塞 decode
- 这是 vLLM 和 SGLang 等系统采用的更先进的策略

**示例（Chunked Prefill，prompt 有 1000 个 token，chunk_size=256）**：
```
Iteration 1: [新请求 prefill token 0-255]   + [现有请求 decode]
Iteration 2: [新请求 prefill token 256-511] + [现有请求 decode]
Iteration 3: [新请求 prefill token 512-767] + [现有请求 decode]
Iteration 4: [新请求 prefill token 768-999] + [现有请求 decode]
Iteration 5: [新请求开始 decode]            + [现有请求 decode]
```

### 4.4 内存管理与 Continuous Batching 的协同

**（1）PagedAttention（分页注意力）**：
- vLLM 的核心创新之一
- 将 KV Cache 按 block 为单位管理，类似操作系统的虚拟内存分页
- 允许 KV Cache 在物理内存中非连续存储
- 当请求完成或被抢占时，可以精确地释放其占用的 block

**（2）Preemption（抢占）**：
- 当 GPU 内存不足以容纳新请求时，可以将某些正在运行的请求暂时"换出"（swap to CPU memory 或 recompute later）

---

## 五、对吞吐量和延迟的影响

### 5.1 吞吐量（Throughput）的提升

- **GPU 利用率最大化**：GPU 几乎始终保持满载运行，不会出现 Static Batching 中 batch 末期的"空转"现象
- 实测数据：相比 Static Batching，吞吐量可提升 **2-8 倍**

**数值对比示例（8 个请求，生成长度分别为 20, 30, 40, 50, 60, 70, 80, 100）**：

| 策略 | 总时间 | 有效利用率 |
|------|--------|---------|
| Static Batching | 100 decode steps | 450/800 = **56.25%** |
| Continuous Batching | 动态 | **接近 100%** |

### 5.2 延迟（Latency）的改善

**（1）首 Token 延迟（TTFT）**
- 新请求不需要等待当前 batch 完成就可以开始处理，TTFT 显著降低

**（2）端到端延迟**
- 已完成的请求立即返回，不需要等待其他请求，短请求延迟大幅改善

**（3）尾延迟（Tail Latency）**
- P99 延迟显著降低，消除了"等待最长请求"的尾延迟问题

### 5.3 吞吐量与延迟的权衡

- batch 越大，吞吐量越高，但单个请求的每 token 生成时间可能略有增加
- 需要根据 SLA 要求在吞吐量和延迟之间取得平衡
- 通常通过设置 `max_batch_size` 和 `max_num_seqs` 等参数来控制

---

## 六、在 vLLM 中的实现

### 6.1 关键组件

**（1）Scheduler（调度器）**
- 位于 `vllm/core/scheduler.py`
- 实现了 iteration-level scheduling
- 每个 step 调用 `scheduler.schedule()` 来决定哪些请求参与本次推理

**（2）SequenceGroup**
- 每个用户请求被封装为一个 SequenceGroup
- 可能包含多个 Sequence（例如 beam search 时一个请求有多个候选序列）
- 状态转移：`WAITING → RUNNING → FINISHED`（或 `SWAPPED`）

**（3）BlockManager（块管理器）**
- 管理 GPU 和 CPU 上的 KV Cache block
- 与 PagedAttention 配合，在请求加入/移出时分配/释放 block

### 6.2 关键配置参数

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    max_num_seqs=256,              # running batch 中的最大序列数
    max_num_batched_tokens=4096,   # 每个 iteration 处理的最大 token 数
    enable_chunked_prefill=True,   # 是否启用 chunked prefill
)
```

---

## 七、在 SGLang 中的实现

### 7.1 RadixAttention

- SGLang 的核心创新
- 使用 Radix Tree（基数树）来管理 KV Cache
- 支持自动前缀共享（Automatic Prefix Caching）
- 当多个请求共享相同的前缀（如系统 prompt）时，可以复用 KV Cache

### 7.2 SGLang 的独特优化

**（1）RadixAttention + Continuous Batching 的协同**：
- 新请求加入 batch 时，先查找 Radix Tree 中是否有可复用的 KV Cache
- 如果有，跳过已缓存部分的 prefill，大幅减少新请求的 prefill 时间

**（2）FlashInfer 后端**：
- SGLang 使用 FlashInfer 作为 attention 计算后端
- 支持 Ragged Tensor 格式，避免 padding 开销，对 variable-length batch 效率高

### 7.3 vLLM vs SGLang 对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| KV Cache 管理 | PagedAttention | RadixAttention |
| 前缀缓存 | 支持（可选） | 原生支持（自动） |
| Chunked Prefill | 支持 | 支持 |
| Attention 后端 | FlashAttention | FlashInfer |
| 调度粒度 | Iteration-level | Iteration-level |

---

## 八、高级话题

### 8.1 Disaggregated Prefill-Decode（PD 分离）

一些先进的系统将 Prefill 和 Decode 分离到不同的 GPU（甚至不同的机器）上：
- **Prefill 节点**：专门处理 prefill，是 compute-bound 任务
- **Decode 节点**：专门处理 decode，是 memory-bound 任务
- 两种节点可以独立做 Continuous Batching，避免互相干扰

### 8.2 Speculative Decoding 与 Continuous Batching 的结合

投机解码可以在 Continuous Batching 框架下运行：
- 每个请求的投机步数可能不同
- 验证步骤中，部分 token 被接受，部分被拒绝
- 调度器需要处理更复杂的状态

---

## 九、总结

Continuous Batching 是现代 LLM 推理引擎的基石技术之一：

1. **核心创新**：从 request-level 调度升级为 iteration-level 调度
2. **关键收益**：吞吐量提升 2-8 倍，延迟显著降低
3. **实现基础**：需要灵活的 KV Cache 管理（如 PagedAttention）配合
4. **行业标准**：vLLM、SGLang、TensorRT-LLM、DeepSpeed 等主流推理框架均已采用
5. **持续演进**：与 Chunked Prefill、PD 分离、投机解码等技术深度融合

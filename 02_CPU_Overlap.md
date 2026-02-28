# 2. CPU Overlap（CPU/GPU 重叠）详解

---

## 一、概念：CPU 与 GPU 计算重叠执行

CPU Overlap（也叫 CPU/GPU Overlap 或 Compute Overlap）指的是让 CPU 上的操作与 GPU 上的操作在时间上重叠执行，从而隐藏其中一方的延迟，提升整体系统吞吐和响应速度。

在大模型推理中，一个完整的推理步骤不仅仅包含 GPU 上的矩阵计算，还包含大量 CPU 端的工作：

- **请求调度（Scheduling）**：决定哪些请求进入当前 batch、优先级排序、抢占策略等
- **Tokenization**：将输入文本编码为 token ID，将输出 token ID 解码为文本
- **采样（Sampling）**：根据 logits 进行 top-k、top-p、temperature 采样，生成下一个 token
- **KV Cache 管理**：分配和释放 KV cache 的内存块、更新页表
- **前处理和后处理**：输入校验、特殊 token 处理、停止条件检查、流式输出等
- **网络 I/O**：接收新请求、发送响应结果

如果这些 CPU 操作和 GPU 计算是串行执行的，那么总延迟 = CPU 时间 + GPU 时间。通过 overlap，可以让 CPU 操作在 GPU 计算的同时进行，使得总延迟约等于 max(CPU 时间, GPU 时间)，从而显著降低端到端延迟。

在 decode 阶段，这一点尤其重要。因为 decode 阶段每步的 GPU 计算时间可能只有几百微秒到几毫秒，而 CPU 端的调度和采样也需要类似量级的时间。如果不做 overlap，CPU 开销可能占据总延迟的 30%-50%。

---

## 二、在推理中的应用

### 2.1 调度与 GPU 计算的重叠

在当前 step 的 GPU 前向计算还在执行时，CPU 就开始为下一个 step 做调度决策：

- 检查是否有新请求到达
- 决定下一个 batch 的组成（哪些请求继续 decode，哪些新请求 prefill）
- 为新请求分配 KV cache 空间
- 计算 attention mask 和 position id

这样当 GPU 前向完成后，下一步的准备工作已经就绪，可以立即开始。

### 2.2 Tokenize/Detokenize 与 GPU 计算的重叠

- **输入 tokenization**：新请求的文本编码可以在当前 GPU step 执行时完成
- **输出 detokenization**：将生成的 token ID 解码为文本可以异步进行，不必等 GPU
- 在流式输出场景下，detokenize 和网络发送可以完全在 GPU 计算的间隙完成

### 2.3 采样（Sampling）与下一步计算的重叠

采样本身可能涉及 GPU 操作（logits 处理在 GPU 上）和 CPU 操作（复杂的采样逻辑、停止条件判断）。一种优化方式是：

- GPU 完成当前 step 的 logits 计算后，将 logits 传回 CPU
- CPU 进行采样的同时，GPU 可以开始做一些不依赖采样结果的预处理工作
- 或者更激进地：用上一步的采样结果开始当前步的 embedding lookup，在上一步采样还在进行时就启动部分 GPU 计算

### 2.4 KV Cache 管理与 GPU 计算的重叠

- 页表更新、内存块分配/释放等操作在 CPU 端完成
- 这些操作可以在 GPU 计算 attention 之前的其他层时并行进行
- Paged Attention 的块管理器（Block Manager）的分配逻辑可以提前执行

### 2.5 网络 I/O 与计算的重叠

- 新请求的接收和解析可以在 GPU 计算时异步进行
- 生成结果的发送（特别是流式场景）不需要等待 GPU 空闲

---

## 三、如何实现

### 3.1 异步调度（Asynchronous Scheduling）

- **核心思想**：将调度器（Scheduler）设计为异步的，它在独立的时间窗口内完成工作
- **vLLM 中的实现**：vLLM 的调度器在每个 step 开始前执行，但通过优化调度逻辑的执行时间，使其尽可能短。更新的版本引入了异步调度，让调度逻辑和 GPU 计算重叠
- **SGLang 中的实现**：SGLang 使用了更激进的 overlap 策略，其 overlap scheduler 可以在 GPU forward 进行的同时完成下一步的调度

### 3.2 多线程/多进程

- 使用专门的线程处理 tokenization/detokenization
- 使用专门的线程处理网络 I/O（asyncio、事件循环）
- 使用专门的线程管理 KV cache 和内存分配
- Python 中受 GIL 限制，通常使用多进程或 C++ 扩展来实现真正的并行
- SGLang 将 tokenizer 放在独立的进程中运行，通过 ZeroMQ 进行通信

### 3.3 Pipeline 流水线

- 将整个推理步骤分解为多个阶段（stage）
- 不同 step 的不同阶段可以流水线化执行
- 例如：step N 的 GPU 计算 | step N 的采样 + step N+1 的调度 | step N+1 的 GPU 计算
- 这类似于 CPU 流水线的思想，通过增加 pipeline 深度来提高吞吐

### 3.4 CUDA Stream 与异步操作

- 利用 CUDA 的异步特性，CPU 在发射 kernel 后可以立即返回做其他事情
- 使用 `cudaEventRecord` 和 `cudaEventQuery` 来检查 GPU 操作是否完成，而不是 `cudaEventSynchronize` 阻塞等待
- 在合适的位置插入同步点，确保数据一致性

### 3.5 具体实现中的技巧

- **提前准备（Prefetch）**：在 GPU 计算时就准备好下一步所需的输入 tensor
- **延迟同步（Lazy Synchronization）**：尽量延迟 CPU-GPU 同步点，让两者各自自由运行
- **双缓冲（Double Buffering）**：使用两套缓冲区轮流工作，一套在 GPU 上处理，另一套在 CPU 端准备

---

## 四、对延迟和吞吐的影响

### 4.1 对延迟的影响

- **单 step 延迟降低**：通过 overlap，每个 step 的总延迟从 `CPU_time + GPU_time` 降低到 `max(CPU_time, GPU_time)`
- **TTFT（Time To First Token）改善**：通过重叠 tokenize 和调度，可以更快地开始 prefill 计算
- **ITL（Inter-Token Latency）改善**：decode 阶段每个 token 的生成间隔缩短
- **典型效果**：在 decode 阶段，overlap 可以降低 20%-40% 的每 token 延迟

### 4.2 对吞吐的影响

- **吞吐直接提升**：减少了每个 step 的总时间，单位时间内可以处理更多 step
- **GPU 利用率提升**：GPU 不需要等待 CPU 完成调度等操作，空闲时间减少
- **CPU 利用率提升**：CPU 不需要等待 GPU 完成计算，可以在等待期间做有用的工作
- **典型效果**：系统吞吐可提升 15%-30%

### 4.3 实际考量

- Overlap 的效果取决于 CPU 和 GPU 工作量的比例。如果 GPU 计算时间远大于 CPU 时间（比如大 batch prefill），overlap 收益有限
- 如果 CPU 成为瓶颈（比如复杂的调度逻辑、大量的 tokenize 工作），即使完美 overlap 也无法完全隐藏 CPU 开销
- 过度的 overlap 可能增加系统复杂度和调试难度
- 内存使用可能增加（双缓冲等策略需要额外空间）

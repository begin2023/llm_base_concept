# SGLang CPU Overlap 调度详解

## 目录

1. [什么是 CPU Overlap](#1-什么是-cpu-overlap)
2. [为什么需要 CPU Overlap](#2-为什么需要-cpu-overlap)
   - 2.1 [Prefill 和 Decode 的完整操作分解（CPU vs GPU）](#21-prefill-和-decode-的完整操作分解)
   - 2.2 [CPU 开销不可忽视](#22-cpu-开销不可忽视)
3. [如何使用](#3-如何使用)
   - 3.3 [环境变量设计目的与原理](#33-相关环境变量)
   - 3.4 [自动禁用场景的详细技术原因](#34-自动禁用的场景)
4. [核心实现原理](#4-核心实现原理)
   - 4.0 [为什么"处理结果₁"和"Forward₂"可以并行？](#40-为什么处理结果和forward可以并行依赖关系分析)
   - 4.1 [双 CUDA Stream 架构](#41-双-cuda-stream-架构)
   - 4.2 [FutureMap：未来值机制](#42-futuremap未来值机制)
   - 4.3 [异步 D2H 拷贝与事件同步](#43-异步-d2h-拷贝与事件同步)
5. [关键数据结构与流程](#5-关键数据结构与流程)
6. [代码走读：一次完整的 Overlap 循环](#6-代码走读一次完整的-overlap-循环)
7. [特殊情况处理](#7-特殊情况处理)
8. [相关的其他 Overlap 机制](#8-相关的其他-overlap-机制)
9. [性能效果](#9-性能效果)
10. [常见问题](#10-常见问题)

---

## 1. 什么是 CPU Overlap

**CPU Overlap Scheduling**（CPU 重叠调度）是 SGLang 推理引擎的核心调度优化。其核心思想是：

> **在 GPU 执行当前 batch 的 forward 计算时，CPU 同时并行处理上一个 batch 的结果（输出处理、请求调度、KV cache 管理等），从而隐藏 CPU 开销，提升整体吞吐。**

### 形象类比

可以类比为**流水线洗碗**：

- **无 Overlap**：洗一个碗 → 擦干 → 放好 → 再洗下一个碗。"擦干"的时候双手空闲。
- **有 Overlap**：洗碗的同时，另一只手把上一个碗擦干放好。两只手同时工作，总时间大幅缩短。

### 时序对比

```
=== 无 Overlap（串行模式）===

时间轴 ──────────────────────────────────────────────────────►

CPU:  [调度 Batch₁] [等待 GPU...] [处理结果₁] [调度 Batch₂] [等待 GPU...] [处理结果₂]
GPU:  [等待 CPU...] [Forward₁   ] [等待 CPU..................] [Forward₂   ] [等待 CPU...]
                                  ↑                           ↑
                              GPU 空闲                     GPU 空闲

=== 有 Overlap（重叠模式）===

时间轴 ──────────────────────────────────────────────────────►

CPU:  [调度 Batch₁] [处理结果₀ + 调度 Batch₂] [处理结果₁ + 调度 Batch₃] ...
GPU:  [Forward₁   ] [Forward₂                ] [Forward₃                ] ...
                    ↑                          ↑
               CPU/GPU 并行！             CPU/GPU 并行！
```

关键差异：Overlap 模式下 **GPU 几乎没有空闲等待 CPU 的气泡（bubble）**。

---

## 2. 为什么需要 CPU Overlap

### 2.1 Prefill 和 Decode 的完整操作分解

要理解 CPU Overlap 的价值，首先需要清楚 LLM 推理的每一步中，哪些操作在 CPU 上执行，哪些在 GPU 上执行：

#### Prefill（首次处理 prompt）阶段

```
请求到达 → 调度 → Forward → 采样 → 输出首 Token
```

| 步骤 | 运行位置 | 具体操作 | 代码位置 |
|------|---------|---------|---------|
| 接收请求 | **CPU** | ZMQ 接收、解析 JSON、tokenize prompt | `scheduler.py: recv_requests()` |
| 请求处理 | **CPU** | 构建 Req 对象、设置 sampling params、Radix Cache 前缀匹配 | `scheduler.py: process_input_requests()` |
| Batch 调度 | **CPU** | 根据策略选择 prefill/decode 混合、确定 batch 组成 | `scheduler.py: get_next_batch_to_run()` |
| Batch 准备 | **CPU+GPU** | 拼接 input_ids（CPU）、分配 KV cache 槽位（GPU 内存操作）、准备 attention metadata | `schedule_batch.py: prepare_for_extend()` |
| Embedding | **GPU** | token → embedding 向量 | `model_runner.py: forward()` |
| Transformer 层 | **GPU** | self-attention + MLP，处理所有 prompt tokens | `model_runner.py: forward()` |
| Logits 计算 | **GPU** | 最后一个 token 的 hidden states → logits (lm_head 矩阵乘) | `model_runner.py: forward()` |
| Logits 预处理 | **GPU** | temperature scaling、top-p/top-k 过滤、repetition penalty | `model_runner.py: _preprocess_logits()` |
| **采样 (Sampling)** | **GPU** | `torch.multinomial` 或 argmax 从 logits 中选择 next token | `model_runner.py: sample()` |
| 结果拷贝 | **GPU→CPU** | next_token_ids、logprobs 等从 GPU 拷贝到 CPU | `utils.py: copy_to_cpu()` |
| 输出处理 | **CPU** | 检查停止条件、detokenize、发送 SSE 流式响应 | `scheduler_output_processor_mixin.py` |
| KV Cache 管理 | **CPU** | 更新 Radix Cache 树结构、释放已完成请求的 cache | `scheduler_output_processor_mixin.py` |

#### Decode（逐 token 生成）阶段

```
准备下一步 → Forward → 采样 → 输出 Token → 循环
```

| 步骤 | 运行位置 | 具体操作 |
|------|---------|---------|
| 接收新请求 | **CPU** | 检查是否有新请求到达 |
| Decode 准备 | **CPU+GPU** | 设置 input_ids（上一步的 output）、分配 1 个新 KV slot | `schedule_batch.py: prepare_for_decode()` |
| Embedding | **GPU** | 单个 token → embedding |
| Transformer 层 | **GPU** | self-attention（利用 KV cache）+ MLP |
| Logits + 预处理 | **GPU** | 同 Prefill |
| **采样** | **GPU** | 同 Prefill，`torch.multinomial` 在 GPU 上执行 |
| 结果拷贝 | **GPU→CPU** | 异步 D2H 拷贝 |
| 输出处理 | **CPU** | detokenize、流式输出、检查 EOS |
| KV Cache 管理 | **CPU** | 释放已完成请求的 cache，为新请求腾空间 |

> **重要结论：采样（Sampling）在 GPU 上执行。** `torch.multinomial` 直接操作 GPU 上的 logits 张量，结果 `next_token_ids` 也在 GPU 上，后续需要 D2H 拷贝到 CPU。

### 2.2 CPU 开销不可忽视

在 decode 阶段，每个 step 的 GPU 计算时间可能只有 **几毫秒到几十毫秒**。而 CPU 需要做的工作并不少：

| CPU 工作项 | 说明 | 耗时量级 |
|-----------|------|---------|
| 接收新请求 | ZMQ recv + 解析 | ~0.1ms |
| 输出处理 | D2H 同步等待 + tensor → Python list | ~0.5-2ms |
| 流式输出 | 检查停止条件、detokenize、发送 SSE 流 | ~0.5-1ms |
| KV Cache 管理 | 释放已完成请求的 KV cache，Radix Cache 树操作 | ~0.1-0.5ms |
| 下一批调度 | 根据策略选择下一个 batch（prefill/decode混合） | ~0.1-0.5ms |
| Batch 准备 | 拼接 input_ids、计算 seq_lens、分配 KV slots | ~0.5-1ms |
| TP 同步 | 多 GPU 间广播调度信息（Gloo all-gather） | ~0.1-0.5ms |

**总计 CPU 开销 ≈ 2-6ms**，而 decode 的 GPU forward 可能只有 5-15ms。如果串行执行，CPU 开销占总时间的 **20-50%**！

CPU Overlap 通过让这些 CPU 工作与 GPU forward 并行，有效隐藏了这部分开销。

---

## 3. 如何使用

### 3.1 默认行为

**CPU Overlap 默认开启，无需额外配置。**

```bash
# 直接启动即可，Overlap 默认生效
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct
```

### 3.2 禁用 Overlap

如果需要调试或在特殊场景下禁用：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --disable-overlap-schedule
```

### 3.3 相关环境变量

#### `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`（默认 `False`）

**设计目的**：优化连续 prefill 场景下的首 token 延迟 (TTFT)。

**问题场景**：当请求突发到达时，可能出现连续两个 batch 都是 prefill 的情况。在 overlap 模式下，Batch₁（prefill）的结果会被延迟到 Batch₂（也是 prefill）的 forward 启动之后才处理。这意味着 Batch₁ 的首个 token 要等到 Batch₂ 开始 forward 后才能发送给用户。

```
=== Overlap ON（默认，TTFT 较高）===

GPU:  [Forward P1      ] [Forward P2      ] [Forward D1       ]
CPU:  [调度 P1         ] [调度 P2 +        [处理 P1 结果 +
                          处理 P0 结果]     调度 D1]
                                            ↑ P1 的首 Token 才在这里发出！

=== 连续 Prefill Overlap 禁用（TTFT 更低）===

GPU:  [Forward P1      ] [idle ] [Forward P2      ] [Forward D1       ]
CPU:  [调度 P1         ] [处理 P1 结果] [调度 P2  ] [处理 P2 + 调度 D1]
                          ↑ P1 的首 Token 在这里就发出了！（更早）
```

**权衡**：改善 TTFT，但引入 GPU idle bubble，略微降低吞吐。只在连续 prefill 场景生效，decode 阶段不受影响。

#### `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH`（默认 `False`）

**设计目的**：控制 DP Attention 模式下多 GPU 调度同步的通信后端。

**背景**：在 DP (Data Parallel) Attention 模式下，多个 DP worker 各自独立调度 batch。但在进入 MLP 共享计算前，所有 worker 需要通过 `all_gather` 同步调度元信息（每个 worker 的 token 数、是否有 prefill、能否用 CUDA Graph 等）。

**默认用 Gloo 的原因**：overlap 模式下，`all_gather` 在 `schedule_stream`（CPU 侧）执行，此时 `forward_stream` 正在 GPU 上跑 forward。如果用 NCCL（走 GPU），all_gather 会与 forward 争抢 GPU 资源。**Gloo 走 CPU + TCP/IP 网络，完全不干扰 GPU 计算**，这正是 overlap 设计的初衷。

```python
# scheduler_dp_attn_mixin.py: prepare_mlp_sync_batch_raw()
if disable_overlap_schedule or NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:
    group = tp_group.device_group    # NCCL（走 GPU）
    device = tp_group.device
else:
    group = tp_group.cpu_group       # Gloo（走 CPU，默认）
    device = "cpu"
```

**何时需要切换到 NCCL**：
- Gloo 在特定网络环境下不稳定或延迟过高
- Forward 计算时间足够长（如大型 prefill），GPU 上的短暂 NCCL 通信不会造成明显 bubble
- 调试排查 Gloo 相关问题时

### 3.4 自动禁用的场景

以下场景中，SGLang 会在 `server_args.py` 中自动将 `disable_overlap_schedule` 设为 `True`：

#### (1) Pipeline Parallelism（`pp_size > 1`）

**代码位置**：`server_args.py:2684`

**技术原因**：PP 有自己独立的事件循环 `event_loop_pp`，通过多个 micro-batch 在 PP stage 间轮转，使用同步的 `send/recv` 进行 rank 间通信。Overlap 调度假设单 stage 模型——一个 `forward_stream` 跑完整个模型并产生 `next_token_ids`。但在 PP 中，只有最后一个 rank 产生 token IDs，前面的 rank 只产出中间 hidden states。**FutureMap 的负数占位符机制无法跨 PP rank 工作**。此外 PP 已通过多 micro-batch 缓冲实现了自己的流水线并行。

#### (2) 投机解码 V1（EAGLE/EAGLE3/STANDALONE，未启用 Spec V2）

**代码位置**：`server_args.py:2797`

**技术原因**：V1 投机解码采用"先 draft 再 verify"的两阶段流程——draft 模型连续生成 N 个候选 token，然后 target 模型一次性验证并接受/拒绝。每个调度步骤包含**多次 GPU forward**（draft + verify），verify 后可能需要回滚 KV cache 中被拒绝的 token。Overlap 调度假设每个调度步骤恰好一次 forward，且结果可以通过 FutureMap 延迟处理。**V1 的 verify 步骤必须同步获得接受/拒绝结果后才能决定下一步调度**，无法延迟。V2 投机解码专门重新设计了与 overlap 兼容的 worker 架构。

#### (3) NGram 投机解码

**代码位置**：`server_args.py:2886`

**技术原因**：NGRAM 投机解码使用 `NGRAMWorker`，该 worker 没有实现 V2 overlap 兼容架构。Draft 阶段虽然是 CPU 端的 N-gram 模式匹配（无需 GPU draft 模型），但 verify 阶段仍需同步获取接受/拒绝结果。如果尝试创建 overlap worker，会直接抛出 `ValueError`。

#### (4) Diffusion LLM

**代码位置**：`server_args.py:3250`

**技术原因**：扩散语言模型的生成范式与自回归模型完全不同——不是逐 token 从左到右生成，而是对整个序列进行多轮迭代去噪。每次 forward 同时精化序列中的**所有 token**，而不是产生单个 `next_token_id`。FutureMap 的"存 next_token_ids → 用负数占位 → 下一轮 resolve"机制在扩散模型中没有意义。扩散 LLM 有自己的 batch 生命周期管理（`dllm/mixin/scheduler.py`），请求经历多轮去噪循环。

#### (5) Mamba 模型 + `no_buffer` 策略 + Radix Cache 启用

**代码位置**：`server_args.py:2070`

**技术原因**：Mamba/SSM 混合模型在 `no_buffer` 策略下，不使用额外缓冲区存储循环状态——状态在需要时从 KV cache/token 历史动态重算。`MambaRadixCache` 要求 `page_size=1`，与 Radix Cache 的驱逐/插入操作紧密耦合。在 overlap 模式下，CPU 在 `schedule_stream` 上处理上一 batch 的结果（释放 KV entries、更新 Radix 树），同时 GPU 在 `forward_stream` 上可能正在读取同一位置的 Mamba 状态——**无 extra buffer 做双缓冲隔离，会导致读写冲突**。`extra_buffer` 策略通过 ping-pong 缓冲区解决了这个问题，此时 overlap 可正常工作。

#### (6) 稀疏 Head Embedding

**代码位置**：`server_args.py:1978`

**技术原因**：`SGLANG_EMBEDDINGS_SPARSE_HEAD` 模式下，模型只计算稀疏子集的 head embedding 输出，产生的是稀疏张量（`torch.sparse`），而非 `next_token_ids`。FutureMap 的整个负数索引/环形缓冲区机制专为 token IDs 设计，无法处理稀疏 embedding 输出。

#### (7) PD-Multiplexing（断言要求）

**代码位置**：`server_args.py:5791`（不是自动禁用，而是 assert 强制要求用户传 `--disable-overlap-schedule`）

**技术原因**：PD-Mux 使用 CUDA Green Context + 流分组，在同一 GPU 上同时运行 prefill 和 decode，并分配独立的 SM 资源（`prefill_stream` + `decode_stream`）。Overlap 调度也使用双流（`schedule_stream` + `forward_stream`），但语义完全不同。**两套双流机制同时运行会产生四个竞争的 CUDA 流**，stream 同步、事件记录、工作分配全部冲突。

#### 动态按 Batch 禁用

除了上述静态禁用，`is_disable_overlap_for_batch()` 还会在运行时逐 batch 判断是否临时禁用 overlap：

| 场景 | 条件 | 原因 |
|------|------|------|
| 连续 Prefill | 前后两个 batch 都是 extend + 环境变量已开启 | 改善 TTFT（见 3.3 节） |
| Spec V2 + Grammar | 当前 batch 用 Spec V2 + 有 Grammar 约束 + 结果队列非空 | Grammar FSM 状态依赖上一步结果更新 vocab mask，延迟处理会导致 mask 过期 |

---

## 4. 核心实现原理

### 4.0 为什么"处理结果₁"和"Forward₂"可以并行？——依赖关系分析

这是理解 CPU Overlap 最关键的问题。直觉上会觉得"Batch₂ 的 input_ids 不是来自 Batch₁ 的采样结果吗？结果还没处理完怎么能开始 Batch₂？"

答案是：**Batch₂ 的 GPU forward 不依赖 Batch₁ 结果的 CPU 处理，只依赖 Batch₁ 的 GPU 采样输出。**

#### 区分两种"结果处理"

"结果处理"实际上包含两个完全不同的阶段：

```
Batch₁ Forward 完成后：

┌────────────────────────────────┐      ┌────────────────────────────────┐
│  阶段 A: GPU 侧结果产出        │      │  阶段 B: CPU 侧结果处理        │
│  (在 forward_stream 上)        │      │  (在 schedule_stream 上)       │
│                                │      │                                │
│  1. logits → sampling          │      │  1. copy_done.synchronize()    │
│  2. 得到 next_token_ids (GPU)  │      │  2. next_token_ids → Python    │
│  3. store_to_map() 存入 buf    │      │  3. 检查 EOS 停止条件          │
│  4. copy_to_cpu() 异步拷贝     │      │  4. detokenize 输出文本        │
│  5. copy_done.record()         │      │  5. 发送 SSE 流式响应          │
│                                │      │  6. 释放已完成请求的 KV cache  │
│  Batch₂ 只依赖这个阶段！       │      │  7. 更新 Radix Cache 树结构    │
│  (通过 FutureMap 解耦)         │      │                                │
└────────────────────────────────┘      │  Batch₂ 完全不依赖这个阶段！   │
                                        └────────────────────────────────┘
```

#### Batch₂ 的 input_ids 如何获取？

关键在于 **FutureMap 的负数占位符机制**将数据依赖从 CPU 转移到了 GPU：

```
时间线：

Step 1: Batch₁ forward 完成（在 forward_stream 上）
        ├── sampling 得到 next_token_ids = [42, 88, 73]  (GPU 张量)
        ├── store_to_map(): 将 [42, 88, 73] 写入 FutureMap 的 token_ids_buf
        └── copy_to_cpu(): 异步拷贝到 CPU（用于后续的 CPU 处理）

Step 2: 回到 schedule_stream，设置 Batch₁ 的 output_ids = [-5, -6, -7]（负数占位符）

Step 3: prepare_for_decode() 将 output_ids 变为 input_ids
        Batch₂ 的 input_ids = [-5, -6, -7]  ← 此时是占位符，不是真实值！

Step 4: Batch₂ forward 启动（在 forward_stream 上）
        ├── resolve_future(): input_ids [-5, -6, -7] → [42, 88, 73]  ← 从 GPU buf 直接读取！
        └── model.forward([42, 88, 73])  ← 使用真实值进行计算

与此同时，CPU 在做 Batch₁ 的"阶段 B"处理（detokenize、流式输出等）
```

**核心洞察**：

1. **GPU→GPU 路径（无需 CPU 中转）**：Batch₁ 的采样结果通过 `store_to_map()` 直接写入 GPU 上的 `token_ids_buf`，Batch₂ 的 `resolve_future()` 直接从 GPU 的 `token_ids_buf` 读取。**整个数据传递全在 GPU 上完成，不经过 CPU**。

2. **CPU 处理只为"人类可见的输出"服务**：`process_batch_result()` 做的事情（detokenize、流式输出、停止条件检查）都是为了把结果返回给用户，对下一个 batch 的 GPU 计算没有任何影响。

3. **唯一例外——Grammar 约束**：Grammar（如 JSON Schema）需要根据上一步输出的 token 更新 vocabulary mask，这确实需要 CPU 处理完毕后才能 sample。这就是为什么 Grammar 场景需要"延迟采样"（见 7.2 节），也是 `is_disable_overlap_for_batch()` 对 Spec V2 + Grammar 做特殊处理的原因。

#### 依赖关系全景图

```
Batch₁ GPU Forward ──────────────────────────┐
  ├── sampling → next_token_ids (GPU)        │
  ├── store_to_map(token_ids_buf)  ──────────┼──► Batch₂ resolve_future()
  ├── copy_to_cpu() (异步)                   │      └── model.forward()
  └── copy_done.record()                     │
                                             │
Batch₁ CPU 处理 ◄── copy_done.synchronize() │    （与 Batch₂ forward 并行执行）
  ├── detokenize                             │
  ├── 流式输出到客户端                        │    ← 这些都不影响 Batch₂
  ├── 检查停止条件                            │
  ├── 释放完成请求的 KV cache                 │
  └── 更新 Radix Cache                       │

Batch₂ 准备（schedule_stream）               │
  ├── get_next_batch_to_run()                │    ← 这些也不依赖 Batch₁ CPU 处理
  ├── prepare_for_decode()                   │
  └── alloc KV slots                         │
```

**为什么 Batch₂ 的调度不依赖 Batch₁ 的结果处理？**

- **KV cache 释放**：Batch₂ 调度时，Batch₁ 中已完成的请求尚未释放 KV cache。但 SGLang 的调度器在 `get_next_batch_to_run()` 中已经预留了足够的内存余量，不需要等上一批释放才能分配。
- **新请求的加入**：`recv_requests()` 和 `process_input_requests()` 是独立的，不依赖上一批的处理结果。
- **Batch 组成**：running batch 中的请求在 `prepare_for_decode()` 时仍然有效（overlap 模式下跳过已 finished 的请求，见 `scheduler_output_processor_mixin.py:409`）。

### 4.1 双 CUDA Stream 架构

Overlap 调度的基础是 **CUDA 多流并行**：

```
                    ┌──────────────────────────────────────────┐
                    │              GPU 设备                      │
                    │                                            │
  CPU Thread ──────►│  schedule_stream (priority=0, 默认流)      │
  (Python)          │    - batch 准备                            │
                    │    - 张量拼接                              │
                    │    - KV cache 分配                         │
                    │                                            │
                    │  forward_stream (专用计算流)               │
                    │    - model forward                         │
                    │    - sampling                              │
                    │    - 结果存入 FutureMap                    │
                    │                                            │
                    │  copy_stream (数据拷贝流)                  │
                    │    - GPU → CPU 异步拷贝 (next_token_ids等) │
                    └──────────────────────────────────────────┘
```

**关键点**：
- `schedule_stream`：CPU 主循环运行在此流上，所有调度操作（接收请求、准备 batch、处理结果）在此执行
- `forward_stream`：GPU forward 计算的专用流。通过 `with self.forward_stream_ctx:` 上下文切换到此流
- `copy_stream`：异步 D2H 拷贝流，将 forward 结果（next_token_ids、logprobs 等）从 GPU 拷回 CPU

### 4.2 FutureMap：未来值机制

这是 Overlap 调度中最巧妙的设计。

**问题**：当 GPU 还在执行 Batch₁ 的 forward 时，CPU 已经需要为 Batch₂ 准备 `input_ids`，但 Batch₂ 的 `input_ids` 来自 Batch₁ 的 sampling 结果——此时结果还没出来！

**解决方案**：使用**负数占位符**（Future Token IDs）。

```
┌─────────────────────────────────────────────────────────────────┐
│                       FutureMap 工作流程                         │
│                                                                  │
│  Step 1: 分配 Future 索引                                        │
│  ┌─────────────────────────────┐                                 │
│  │ alloc_future_indices(bs=3)  │                                 │
│  │ → indices = [5, 6, 7]      │  ← 环形缓冲区中的位置            │
│  │ → 返回 FutureIndices       │                                 │
│  └─────────────────────────────┘                                 │
│                                                                  │
│  Step 2: Batch₂ 的 input_ids 设为负数占位符                      │
│  ┌─────────────────────────────┐                                 │
│  │ batch.output_ids = [-5, -6, -7]                               │
│  │ (prepare_for_decode 时变为 input_ids)                         │
│  └─────────────────────────────┘                                 │
│                                                                  │
│  Step 3: Batch₁ forward 完成后，存入真实值                        │
│  ┌─────────────────────────────┐                                 │
│  │ store_to_map(indices, result)                                 │
│  │ token_ids_buf[5] = 12345  ← 真实 token ID                    │
│  │ token_ids_buf[6] = 67890                                      │
│  │ token_ids_buf[7] = 11111                                      │
│  └─────────────────────────────┘                                 │
│                                                                  │
│  Step 4: Batch₂ forward 开始前，解析占位符                        │
│  ┌─────────────────────────────┐                                 │
│  │ resolve_future(batch₂)                                        │
│  │ input_ids: [-5, -6, -7]                                       │
│  │         → [ 12345, 67890, 11111]  ← 从 buf 中查找替换         │
│  └─────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

`_resolve_future_token_ids` 的实现（使用 `torch.compile` 加速）：

```python
@torch.compile(dynamic=True, backend=get_compiler_backend())
def _resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,                                    # 负数 = 占位符
        future_token_ids_map[torch.clamp(-input_ids, min=0)],  # 从 buf 取真实值
        input_ids,                                        # 正数 = 已知值，保持不变
    )
```

### 4.3 异步 D2H 拷贝与事件同步

Forward 完成后，结果需要从 GPU 拷到 CPU 用于后续处理：

```python
# 在 forward_stream 上执行
batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
batch_result.copy_done.record()  # 记录一个 CUDA Event
```

`copy_to_cpu` 使用 `non_blocking=True` 异步拷贝：

```python
def copy_to_cpu(self, return_logprob: bool):
    self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)
    if return_logprob:
        self.logits_output.next_token_logprobs = \
            self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
        # ... 其他 logprob 张量的异步拷贝
    self.copy_done.record()  # 记录完成事件
```

在处理结果时，通过 `copy_done.synchronize()` 等待拷贝完成：

```python
def process_batch_result_decode(self, batch, result):
    if result.copy_done is not None:
        result.copy_done.synchronize()  # 此处才真正阻塞等待
    # ... 后续处理
```

理想情况下，CPU 的调度工作足够填满 GPU forward 的时间，`synchronize()` 时数据已经就绪，无需等待。

---

## 5. 关键数据结构与流程

### 5.1 核心类一览

| 类/函数 | 文件 | 职责 |
|---------|------|------|
| `Scheduler.event_loop_overlap()` | `managers/scheduler.py` | Overlap 主事件循环 |
| `Scheduler.init_overlap()` | `managers/scheduler.py` | 初始化 streams 和 FutureMap |
| `Scheduler.run_batch()` | `managers/scheduler.py` | 在 forward_stream 上启动 forward |
| `Scheduler.record_batch_in_overlap()` | `managers/scheduler.py` | 防止 GPU 张量被 Python GC 回收 |
| `FutureMap` | `managers/overlap_utils.py` | 环形缓冲区，管理 future token IDs |
| `FutureMap.resolve_future()` | `managers/overlap_utils.py` | 将负数占位符替换为真实值 |
| `FutureMap.store_to_map()` | `managers/overlap_utils.py` | 将 forward 结果存入缓冲区 |
| `GenerationBatchResult.copy_to_cpu()` | `managers/utils.py` | 异步 D2H 拷贝 |

### 5.2 FutureMap 环形缓冲区

```
        future_ct (写指针)
            ↓
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │...│ N │  token_ids_buf
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
          ↑                   ↑
     上一次的结果         当前分配的 future 区间

缓冲区大小 = max_running_requests × (3 + max_num_chunks) + 2 × max_running_requests
```

---

## 6. 代码走读：一次完整的 Overlap 循环

以 `event_loop_overlap()` 为入口，跟踪一个完整的 overlap 迭代：

```python
# ==================== event_loop_overlap ====================
# 文件: python/sglang/srt/managers/scheduler.py

def event_loop_overlap(self):
    self.result_queue = deque()  # 存放 (batch, result) 的队列

    while True:
        # ① 接收新请求（CPU 工作）
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # ② 调度下一个 batch（CPU 工作）
        batch = self.get_next_batch_to_run()
        disable_overlap = self.is_disable_overlap_for_batch(batch)

        # ③ 如果不需要 overlap（如连续 prefill），先处理上一个结果
        if disable_overlap:
            pop_and_process()   # 同步处理上一批结果

        # ④ 启动当前 batch 的 forward（非阻塞！）
        if batch:
            batch_result = self.run_batch(batch)          # → forward_stream
            self.result_queue.append((batch.copy(), batch_result))

        # ⑤ 处理上一个 batch 的结果（与 ④ 的 GPU forward 并行！）
        if self.last_batch and not disable_overlap:
            pop_and_process()   # CPU 处理 + GPU forward 并行执行

        # ⑥ 延迟采样（Grammar 场景）
        self.launch_batch_sample_if_needed(batch_result)

        self.last_batch = batch
```

### run_batch 的 Overlap 分支

```python
# 文件: python/sglang/srt/managers/scheduler.py

def run_batch(self, batch):
    if self.enable_overlap:
        model_worker_batch = batch.get_model_worker_batch()

        # (a) 保持引用，防止 GC 回收 GPU 张量
        self.record_batch_in_overlap(model_worker_batch)

        # (b) 复制 sampling_info（forward 中会修改）
        model_worker_batch.sampling_info = \
            model_worker_batch.sampling_info.copy_for_forward()

        # (c) 分配 future 索引
        future_indices = self.future_map.alloc_future_indices(bs)

        # (d) 在 forward_stream 上启动 forward
        with self.forward_stream_ctx:
            # 等待 schedule_stream 的工作完成
            self.forward_stream.wait_stream(self.schedule_stream)

            # 解析上一轮的占位符 → 真实 token IDs
            self.future_map.resolve_future(model_worker_batch)

            # 执行 model forward + sampling
            batch_result = self.model_worker.forward_batch_generation(
                model_worker_batch
            )

            # 将结果存入 FutureMap，供下一轮使用
            self.future_map.store_to_map(future_indices, batch_result)

            # 异步拷贝结果到 CPU
            batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
            batch_result.copy_done.record()

        # (e) 设置负数占位符作为下一轮的 input_ids
        batch.output_ids = -future_indices.indices
```

### 时序图

```
schedule_stream (CPU)          forward_stream (GPU)
       │                              │
  [调度 Batch₂]                       │
  [alloc_future(bs)]                  │
       │                              │
       │──── forward_stream_ctx ─────►│
       │                              │ wait_stream(schedule_stream)
       │                              │ resolve_future(Batch₂)     ← 替换占位符
       │                              │ model.forward(Batch₂)      ← GPU 计算
       │                              │ sample(logits)
       │                              │ store_to_map(result)       ← 存结果
       │                              │ copy_to_cpu()              ← 异步 D2H
       │◄───── 返回 ─────────────────│ copy_done.record()
       │                              │
  [pop_and_process(Batch₁)]          │ ← CPU 处理与 GPU 计算并行！
       │ copy_done.synchronize()      │
       │ process result...            │
       │ free KV cache...             │
       │ stream output...             │
       │                              │
  [调度 Batch₃]                       │
       ...                            ...
```

---

## 7. 特殊情况处理

### 7.1 连续 Prefill 优化

当两个连续的 batch 都是 prefill（extend）时，overlap 会导致第一个 prefill 的输出延迟增加（因为结果要等到下一轮才处理）。SGLang 提供了环境变量来禁用此场景的 overlap：

```python
# is_disable_overlap_for_batch()
disable = (
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
    and batch.forward_mode.is_extend()
    and self.last_batch.forward_mode.is_extend()
)
```

设置 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1` 可改善 TTFT（首 token 延迟），但可能略微降低吞吐。

### 7.2 Grammar 约束的延迟采样

Grammar（如 JSON Schema 约束）需要根据上一步的 token 更新 vocabulary mask。在 overlap 模式下，上一步结果在 forward 开始时可能还未处理完。因此采用**延迟采样**：

```python
# tp_worker.py
if self.enable_overlap and sampling_info.grammars is not None:
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(
            logits_output, forward_batch
        )
        return batch_result

    batch_result.delay_sample_func = sample_batch_func
    return batch_result  # 先返回，稍后再 sample
```

采样被推迟到 `launch_batch_sample_if_needed()` 中执行，此时上一轮结果已处理完毕，grammar mask 已更新。

### 7.3 避免原地操作

在 overlap 模式下，`schedule_stream` 上的张量可能同时被 `forward_stream` 读取。因此必须避免原地修改：

```python
# schedule_batch.py - prepare_for_decode()
# ❌ 错误：原地操作会破坏 forward_stream 正在读取的数据
# self.seq_lens += 1

# ✅ 正确：创建新张量
self.seq_lens = self.seq_lens + 1
```

### 7.4 GC 保护

Overlap 模式下，Python GC 可能在 GPU 还在读取张量时回收 CPU 侧的引用，导致 CUDA 内存被释放。`record_batch_in_overlap()` 通过维持引用来防止这种情况：

```python
def record_batch_in_overlap(self, model_worker_batch):
    # 双缓冲：保持最近两个 batch 的引用
    self.batch_record_ct = (self.batch_record_ct + 1) % 2
    self.batch_record_buf[self.batch_record_ct] = model_worker_batch
```

### 7.5 Decode 的 seq_lens 延迟

由于 overlap 模式下 output_ids 有一步延迟（使用负数占位符），`prepare_for_decode` 中计算 prefix_lens 时需要调整：

```python
# 有 overlap 时 delta = 0，无 overlap 时 delta = -1
delta = 0 if self.enable_overlap else -1

self.prefix_lens.extend([
    len(r.origin_input_ids) + len(r.output_ids) + delta
    for r in running_batch.reqs
])
```

---

## 8. 相关的其他 Overlap 机制

SGLang 中还有两种更高级的 overlap 机制，与 CPU Overlap Schedule 属于不同层面：

### 8.1 Two-Batch Overlap (TBO)

**目标**：在 MoE 模型中，将一个 batch 拆成两个 micro-batch，通过流水线方式交错执行不同层的计算，最大化 GPU 利用率。

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-two-batch-overlap
```

**原理**：将模型的每一层拆分为多个 stage（如 attention → dispatch → experts → combine），两个 micro-batch 错开执行：

```
Batch A:  [Attn] [Dispatch] [Experts] [Combine]
Batch B:         [Attn    ] [Dispatch] [Experts] [Combine]
                 ↑ 利用 A 做 Dispatch 时的空闲计算资源
```

主要用于 DeepSeek V2/V3、Qwen3 MoE 等大型 MoE 模型。

### 8.2 Single-Batch Overlap (SBO)

**目标**：在单个 micro-batch 内部，将通信操作（如 DeepEP 的 combine）与计算操作（如 down-projection GEMM）重叠。

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-single-batch-overlap
```

### 8.3 LoRA Overlap Loading

异步加载 LoRA adapter 权重，将 H2D 传输与 GPU 计算重叠：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --enable-lora-overlap-loading
```

### 8.4 层级关系

```
┌─────────────────────────────────────────────────┐
│            SGLang Overlap 机制层级               │
│                                                  │
│  Level 1: CPU Overlap Schedule     (默认开启)    │
│  ├── CPU 调度 与 GPU forward 并行               │
│  ├── 适用于所有模型                              │
│  └── 提升整体吞吐，降低 inter-token latency     │
│                                                  │
│  Level 2: Two-Batch Overlap (TBO)  (手动开启)    │
│  ├── 两个 micro-batch 交错流水线执行            │
│  ├── 主要用于 MoE + EP 模型                     │
│  └── 提升 GPU 计算利用率                         │
│                                                  │
│  Level 3: Single-Batch Overlap (SBO) (手动开启)  │
│  ├── 单 batch 内通信与计算重叠                  │
│  ├── 主要用于 DeepEP 场景                       │
│  └── 隐藏通信延迟                               │
│                                                  │
│  LoRA Overlap Loading              (手动开启)    │
│  ├── LoRA 权重加载与 forward 并行               │
│  └── 适用于多 LoRA 服务场景                     │
│                                                  │
│  三个 Level 可以组合使用，互不冲突                │
└─────────────────────────────────────────────────┘
```

---

## 9. 性能效果

### 9.1 CPU Overlap 的核心收益

CPU Overlap Schedule 主要在 **decode 阶段** 产生收益，因为 decode 的 GPU forward 时间较短（几毫秒级），CPU 开销占比更大。具体表现为：

- **降低 inter-token latency**：隐藏 CPU 调度/处理开销
- **提升吞吐 (tokens/s)**：减少 GPU 空闲 bubble
- **对高并发场景效果更明显**：batch size 越大，CPU 处理工作越多，overlap 的收益越大

### 9.2 Bubble 监控

SGLang 提供了 bubble metric 来衡量 overlap 的效果：

- `bubble_time`：GPU forward 开始前，等待 `schedule_stream` 完成的时间
- 理想情况下 bubble_time ≈ 0，说明 CPU 工作完全被 GPU forward 隐藏

### 9.3 参考数据

根据 SGLang v0.4 的发布博客（2024-12-04），overlap scheduling 是 v0.4 版本的关键优化之一，与其他优化（如 RadixAttention、chunked prefill）共同使 SGLang 在多个 benchmark 上达到 SOTA 性能。

---

## 10. 常见问题

### Q1: 为什么 output_ids 是负数？

负数是 FutureMap 的占位符。在 overlap 模式下，当前 batch 的 output_ids 在 GPU forward 完成前就需要设置，所以使用 `-future_indices` 作为占位符。在下一轮 forward 开始前，`resolve_future()` 会将这些负数替换为真实的 token IDs。

### Q2: Overlap 会影响输出正确性吗？

不会。Overlap 只改变了 CPU 和 GPU 工作的时序，不影响计算结果。`forward_stream.wait_stream(schedule_stream)` 和 `copy_done.synchronize()` 保证了数据一致性。

### Q3: 什么时候应该禁用 Overlap？

通常不需要禁用。以下场景可能需要：
- 调试时想简化执行流程
- 极低延迟要求且 batch size 很小（CPU 开销本身很小）
- 使用的功能组合与 overlap 不兼容（SGLang 会自动处理大部分情况）

### Q4: Overlap 和 CUDA Graph 兼容吗？

兼容。Overlap 工作在 scheduler 层面（多流调度），CUDA Graph 工作在 model forward 层面（图捕获），两者互不冲突。

### Q5: TBO 和 CPU Overlap 有什么区别？

- **CPU Overlap**：调度/处理（CPU）与 forward（GPU）并行，**跨 batch** 重叠
- **TBO**：两个 micro-batch **同一 forward 内** 交错执行模型层，充分利用 GPU 内的计算和通信资源

两者可以同时开启。


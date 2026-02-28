# 29. Chunked Prefill（分块预填充）

---

## 一、背景：Prefill 与 Decode 的冲突

在 LLM 推理的 Continuous Batching 框架中，Prefill（预填充）和 Decode（解码）阶段在同一 GPU 上执行，存在严重的干扰问题：

**Prefill 的特性**：
- 需要处理整个 prompt（可能有数千 token）
- 是 compute-bound（计算密集型）
- 单次 Prefill 可能耗时 100ms - 数秒（取决于 prompt 长度）

**Decode 的特性**：
- 每次只处理一个 token
- 是 memory-bound（内存带宽密集型）
- 需要低且稳定的延迟（目标 TPOT < 50ms）

**冲突**：当系统需要为新请求做 Prefill 时，正在进行 Decode 的请求必须等待 Prefill 完成，导致 TPOT 大幅增大（尾延迟恶化）。

---

## 二、Chunked Prefill 的核心思想

Chunked Prefill 将一个长 Prefill 任务拆分成多个固定大小的 chunk（块），每次 iteration 只处理一个 chunk 的 Prefill，剩余 token 留到下一次 iteration 处理。

```
不使用 Chunked Prefill：
Iteration 1: [Prefill(1000 tokens)] → 耗时很长，所有 Decode 请求等待
Iteration 2: [Decode]
Iteration 3: [Decode]

使用 Chunked Prefill（chunk_size=256）：
Iteration 1: [Prefill chunk 1(256 tokens)] + [Decode]
Iteration 2: [Prefill chunk 2(256 tokens)] + [Decode]
Iteration 3: [Prefill chunk 3(256 tokens)] + [Decode]
Iteration 4: [Prefill chunk 4(256 tokens)] + [Decode]
Iteration 5: [Decode]（新请求进入 decode 阶段）+ [Decode]
```

每次 iteration 的总 token 数被限制在 `max_num_batched_tokens`（如 1024）以内。

---

## 三、Chunked Prefill 的详细机制

### 3.1 Prefill 的分块

设 prompt 长度为 L，chunk_size 为 C：
- 第 1 次 iteration：处理 token[0:C]，生成并缓存对应 KV Cache
- 第 2 次 iteration：处理 token[C:2C]，此时 token[0:C] 的 KV Cache 已在 GPU 上
- ...
- 第 ⌈L/C⌉ 次 iteration：处理最后一个 chunk，Prefill 完成，开始 Decode

**注意**：Prefill 的 Attention 是双向的（prompt 内部），每个 chunk 内的 token 需要 attend 到之前所有已处理的 KV Cache。

### 3.2 混合 Batch

在 Chunked Prefill 模式下，每个 iteration 的 batch 可能包含：
- 若干个 Prefill chunk（来自不同的新请求）
- 若干个 Decode token（来自正在进行 decode 的请求）

这种混合 batch 需要特殊的 Attention mask 处理：
- Prefill 部分：可以 attend 到所有之前的 token（包括自己的 prompt 前缀）
- Decode 部分：只能 attend 到自己的 KV Cache（不能 attend 到 Prefill 的 token）

```
混合 batch 的 attention mask 示例：

batch tokens: [P0, P1, P2, D0, D1]
（P0-P2 是某请求的 Prefill chunk 3 个 token，D0-D1 是两个 Decode token）

Attention mask:
     P0   P1   P2   D0   D1
P0 [  1    0    0    0    0 ]
P1 [  1    1    0    0    0 ]
P2 [  1    1    1    0    0 ]
D0 [  0    0    0    1    0 ]  ← D0 只能 attend 到自己之前的 KV Cache
D1 [  0    0    0    0    1 ]  ← D1 同理
```

---

## 四、Chunked Prefill 的效果

### 4.1 稳定 TPOT（每 Token 延迟）

```
不使用 Chunked Prefill 时的 TPOT 时序：
时间：  1ms  2ms  3ms  4ms  ... 500ms  501ms  502ms
事件：  D    D    D    D    ...   P      D      D
TPOT：  1ms  1ms  1ms  1ms  ... 500ms   1ms    1ms
       ← 正常 →              ← 尖峰 →  ← 正常 →
（P=Prefill 500ms，D=Decode 1ms）

使用 Chunked Prefill（chunk=50ms）后：
时间：  1ms  51ms  101ms  151ms  201ms  ...
事件：  D+P  D+P   D+P    D+P    D      ...
TPOT：  51ms 51ms  51ms   51ms   1ms    ...
← 平滑，无大尖峰 →
```

### 4.2 Trade-off 分析

Chunked Prefill 引入了新的权衡：

**优化项**：
- Decode 请求的 TPOT 更稳定（P99 大幅降低）
- 减少因 Prefill 导致的 Decode 阻塞

**代价**：
- TTFT 增加：新请求的 Prefill 被分成多个 iteration，首 token 生成时间增大
  - 不使用 Chunked Prefill：TTFT = 1 个 Prefill 时间
  - 使用 Chunked Prefill：TTFT = ⌈L/C⌉ 个 iteration 时间 > 1 个 Prefill 时间
- 计算效率略有下降：小 chunk 的矩阵乘法不如大矩阵高效（GPU 利用率略低）

### 4.3 理想的 Chunk Size

Chunk size 的选择是超参数优化问题：

- **太小**（如 32 tokens）：每个 chunk 计算量小，GPU 利用率低；TTFT 过高
- **太大**（如 2048 tokens）：相当于不使用 Chunked Prefill，TPOT 抖动大
- **推荐范围**：256 - 1024 tokens，根据实际 TTFT 和 TPOT SLA 调整

---

## 五、在 vLLM 和 SGLang 中的配置

### 5.1 vLLM

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,  # 每次 iteration 最多处理 2048 tokens
    # max_num_seqs=256,           # 最多同时处理的请求数
)
```

命令行：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```

**vLLM 的相关参数**：
- `enable_chunked_prefill`：是否启用 Chunked Prefill
- `max_num_batched_tokens`：每次 iteration 处理的最大 token 数（控制 chunk 大小）
- `max_num_seqs`：running batch 中的最大请求数

### 5.2 SGLang

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --chunked-prefill-size 512 \  # chunk 大小
    --max-running-requests 256
```

---

## 六、Chunked Prefill 与其他技术的结合

### 6.1 Chunked Prefill + Prefix Caching

当启用 Prefix Caching 时，Chunked Prefill 只需处理未缓存的 chunk：

```
Prompt = [cached_prefix(1000 tokens)] + [new_content(500 tokens)]

不使用 Prefix Caching：需要 Prefill 1500 tokens
使用 Prefix Caching：只需 Prefill 500 tokens
使用 Prefix Caching + Chunked Prefill（chunk=256）：
  - Iteration 1: chunk[0:256] + Decode
  - Iteration 2: chunk[256:500] + Decode
  → TTFT 进一步降低
```

### 6.2 Chunked Prefill + PD 分离

在 PD 分离架构中，Chunked Prefill 变得不那么必要（因为 P 节点和 D 节点已经分离），但仍可用于控制 P 节点的资源消耗。

### 6.3 Chunked Prefill + Speculative Decoding

Chunked Prefill 与 Speculative Decoding 可以同时启用，但调度器需要处理更复杂的混合 batch（Prefill chunk + Speculative Decode）。

---

## 七、Sarathi-Serve：Chunked Prefill 的研究

Sarathi-Serve（Agrawal et al., 2023）是最早系统研究 Chunked Prefill 的工作：

- 提出 "Stall-free Batching"：通过 Chunked Prefill 消除 Decode 的停顿
- 量化了 Chunked Prefill 对 TTFT 和 TPOT 的 trade-off
- 结论：Chunked Prefill 可以将 Decode 的 P99 延迟降低 5-10×，TTFT 代价可接受

---

## 八、总结

Chunked Prefill 的核心价值：

1. **问题**：长 Prefill 阻塞 Decode，导致 TPOT 尖峰和尾延迟恶化
2. **方案**：将 Prefill 拆分成小 chunk，与 Decode 交替执行
3. **收益**：Decode TPOT 更稳定，P99 延迟大幅降低
4. **代价**：TTFT 略有增加，需要根据 SLA 调整 chunk size
5. **适用场景**：对 TPOT 稳定性要求高（流式输出体验好）的在线服务
6. **框架支持**：vLLM 和 SGLang 均已原生支持

# 18. vLLM 对比 SGLang

---

## 一、背景：两大主流推理框架

vLLM 和 SGLang 是目前 LLM 推理领域最主流的两个开源框架。

- **vLLM**：2023 年由 UC Berkeley 发布，论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》，以 PagedAttention 为核心创新，是目前使用最广泛的推理框架
- **SGLang**：2024 年由 Lianmin Zheng（vLLM 原作者之一）等人发布，论文《SGLang: Efficient Execution of Structured Language Model Programs》，以 RadixAttention 和高效结构化输出为核心优势

---

## 二、核心架构对比

### 2.1 系统架构

**vLLM 架构：**
```
客户端请求
    ↓
AsyncLLMEngine（异步引擎）
    ↓
Scheduler（调度器）
    ├── BlockManager（KV Cache 块管理）
    └── WaitingQueue / RunningQueue
    ↓
Worker（每个 GPU 一个 Worker）
    ├── ModelRunner（模型执行）
    └── CacheEngine（KV Cache 操作）
    ↓
Model（Transformer 前向）
    └── Attention（PagedAttention）
```

**SGLang 架构：**
```
客户端请求
    ↓
Router（多节点路由，可选）
    ↓
TokenizerManager（Tokenizer 管理）
    ↓
Scheduler（调度器）
    ├── RadixCache（KV Cache 基数树管理）
    └── WaitingQueue / RunningBatch
    ↓
ModelRPC（模型 RPC 调用）
    ↓
Model（Transformer 前向）
    └── Attention（FlashInfer）
```

### 2.2 核心组件对比

| 组件 | vLLM | SGLang |
|------|------|--------|
| KV Cache 管理 | PagedAttention（块管理） | RadixAttention（基数树） |
| Attention 后端 | FlashAttention-2 | FlashInfer |
| 前缀缓存 | Prefix Caching（可选启用） | 原生自动前缀缓存 |
| 结构化输出 | Outlines/Guidance | XGrammar + 原生集成 |
| 多节点支持 | 通过 Ray | 原生 RPC + Router |
| 调度粒度 | Iteration-level | Iteration-level |
| 并行策略 | TP/PP/EP | TP/EP，实验性 PP |

---

## 三、KV Cache 管理：PagedAttention vs RadixAttention

这是两个框架最核心的差异。

### 3.1 PagedAttention（vLLM）

**原理**：将 KV Cache 分为固定大小的 block（默认 16 tokens/block），类似操作系统的内存分页：
- 每个请求的 KV Cache 由多个非连续 block 组成
- 使用 block table 记录逻辑 block 到物理 block 的映射
- 支持 Copy-on-Write（写时复制），用于 beam search 等场景

**前缀缓存（Prefix Caching）**：
- vLLM 0.4+ 支持 Automatic Prefix Caching（APC）
- 通过 hash 来标识 block，相同内容的 block 可以复用
- 但需要显式启用（`--enable-prefix-caching`）

### 3.2 RadixAttention（SGLang）

**原理**：使用 Radix Tree（基数树）来管理所有请求的 KV Cache：
- Radix Tree 的每条边对应一段 token 序列
- 树的每个节点对应一段 KV Cache
- 新请求到来时，先在树中查找最长公共前缀，复用已有 KV Cache

**核心优势**：
- **自动前缀复用**：无需额外配置，所有请求自动享受前缀缓存
- **更细粒度**：前缀复用粒度可以是 token 级别（而非 block 级别）
- **多请求共享系统 prompt**：对于 system prompt 相同的场景，性能提升显著

**示例**：
```
已有缓存：
  "You are a helpful assistant. User: What is Python?"
  "You are a helpful assistant. User: What is Java?"

新请求：
  "You are a helpful assistant. User: What is C++?"
  → RadixAttention 自动复用 "You are a helpful assistant. User: " 的 KV Cache
  → 只需计算 "What is C++?" 的 KV Cache
```

---

## 四、Attention 后端：FlashAttention vs FlashInfer

### 4.1 FlashAttention（vLLM 主要后端）

- FlashAttention-2 是目前最主流的 Attention 实现
- 针对固定 batch size 和序列长度优化
- 对 prefill（dense attention）效率高
- 对 variable-length batch 处理较为复杂

### 4.2 FlashInfer（SGLang 主要后端）

FlashInfer 专为推理场景设计：
- **Ragged Tensor 支持**：原生支持变长序列，无需 padding
- **PagedKV Cache 集成**：直接支持分页 KV Cache 的 attention 计算
- **多种 attention 变体**：支持 prefill、decode、mixed（prefill+decode 混合）等多种模式
- **性能优化**：在 decode 阶段（小 batch、长 KV sequence）性能通常优于 FlashAttention

SGLang 的吞吐量优势部分来自于 FlashInfer 在 decode 阶段的效率。

---

## 五、性能对比

### 5.1 吞吐量（Throughput）

**总体结论（2024 年测试数据）**：
- SGLang 在大多数场景下吞吐量高于 vLLM，差距约 10%-50%
- 差距来源：FlashInfer 效率更高、RadixAttention 减少重复计算、更优化的调度

**有前缀缓存场景**：
- SGLang 优势更明显，因为 RadixAttention 自动复用，而 vLLM 需要显式启用
- 有大量共享 system prompt 的场景：SGLang 吞吐量可比 vLLM 高 2-3 倍

**无前缀缓存场景**：
- 差距较小，主要来自 FlashInfer vs FlashAttention 的效率差异

### 5.2 延迟（Latency）

- **TTFT（首 token 延迟）**：相近，两者都支持 chunked prefill
- **TPOT（每 token 生成时间）**：SGLang 通常略低
- **端到端延迟**：相近，SGLang 在高并发场景略有优势

### 5.3 官方 Benchmark 数据（仅供参考，会随版本更新变化）

| 场景 | vLLM | SGLang |
|------|------|--------|
| Llama-3 8B, A100, no prefix | ~2000 tokens/s | ~2500 tokens/s |
| Llama-3 8B, A100, with prefix | ~1500 tokens/s | ~3500 tokens/s |
| Mixtral 8x7B | ~800 tokens/s | ~1000 tokens/s |

---

## 六、功能特性对比

| 功能 | vLLM | SGLang |
|------|------|--------|
| **模型支持** | 非常广泛（100+ 模型） | 较广泛，主流模型均支持 |
| **量化支持** | AWQ, GPTQ, FP8, BnB | AWQ, GPTQ, FP8 |
| **多模态** | LLaVA, Qwen-VL 等 | LLaVA, Qwen-VL 等 |
| **Speculative Decoding** | 支持（多种策略） | 支持 |
| **LoRA** | 支持多 LoRA 并发 | 支持 |
| **结构化输出** | Outlines（外部集成） | XGrammar（深度集成） |
| **Embedding** | 支持 | 支持 |
| **PD 分离** | 实验性支持 | 支持（更完善） |
| **API 兼容性** | OpenAI 兼容 | OpenAI 兼容 |
| **生产稳定性** | 高（大量生产验证） | 较高（快速迭代中） |
| **社区生态** | 非常活跃（Star 最多） | 活跃 |
| **文档质量** | 良好 | 良好 |

---

## 七、工程实践差异

### 7.1 部署方式

**vLLM**：
```bash
# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9
```

**SGLang**：
```bash
# 启动 SGLang 服务
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --tp 4 \
    --mem-fraction-static 0.85
```

### 7.2 调度器设计差异

**vLLM 调度器特点**：
- 有明确的 WAITING / RUNNING / SWAPPED 三态
- 支持 preemption（抢占）：内存不足时将请求 swap 到 CPU
- 支持 recompute 和 swap 两种抢占策略

**SGLang 调度器特点**：
- 更简洁，主要靠 RadixCache 来管理内存
- 当内存不足时，evict（驱逐）最久未使用的 KV Cache 节点
- 类似 LRU Cache 的管理策略

### 7.3 多 LoRA 支持

- **vLLM**：支持 Multi-LoRA serving，可同时为多个 LoRA adapter 提供服务，动态切换
- **SGLang**：也支持 LoRA，但 Multi-LoRA 的成熟度稍低于 vLLM

---

## 八、选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 生产环境，稳定性优先 | vLLM | 更成熟、更多生产验证 |
| 追求最高吞吐量 | SGLang | FlashInfer + RadixAttention 性能更优 |
| 有大量相同 system prompt 的请求 | SGLang | RadixAttention 自动前缀复用 |
| 需要结构化输出（复杂 schema） | SGLang | XGrammar 性能更好 |
| 需要支持很多模型 | vLLM | 模型支持更广泛 |
| 多 LoRA 并发 | vLLM | Multi-LoRA 更成熟 |
| 学术研究 / 快速试验 | SGLang | 更简洁的 API，更快的创新 |
| 企业级部署 | vLLM | 更完善的文档和社区支持 |

---

## 九、总结

vLLM 和 SGLang 代表了 LLM 推理框架的两条技术路线：

1. **vLLM**：以 PagedAttention 为核心，更注重通用性、稳定性和生态
2. **SGLang**：以 RadixAttention + FlashInfer 为核心，更注重性能极限和结构化生成

两者都在快速迭代，差距在缩小。实际选择时应根据具体业务场景、稳定性需求和性能目标综合考量。从技术趋势看，两者的优秀设计会相互借鉴：vLLM 已引入类似 RadixAttention 的前缀缓存，SGLang 也在不断完善工程化能力。

# 28. Prefix Caching / Radix Attention

---

## 一、什么是前缀缓存

前缀缓存（Prefix Caching）是一种 KV Cache 复用技术：当多个请求共享相同的前缀（prefix）时，只计算一次该前缀的 KV Cache，后续所有共享该前缀的请求直接复用，而不重新计算。

**典型共享场景**：

1. **System Prompt 共享**：同一应用的所有请求都使用相同的 system prompt（如"你是一个专业的客服助手..."）
2. **Few-shot 示例共享**：所有请求携带相同的 few-shot examples
3. **长文档 QA**：同一文档的多个问题，文档内容（前缀）相同
4. **多轮对话**：历史对话记录作为前缀，新问题附在后面
5. **代码补全**：同一代码文件的多个补全请求

---

## 二、Prefix Caching 的工作原理（vLLM）

### 2.1 基于 Hash 的 Block 识别

vLLM 的 Automatic Prefix Caching（APC）基于 KV Cache block 级别的 hash：

```
KV Cache 按 block 管理（默认每 block 16 tokens）

Block 0: token[0:16]   → hash = H(token[0:16])
Block 1: token[16:32]  → hash = H(token[0:16], token[16:32])  # 级联 hash
Block 2: token[32:48]  → hash = H(token[0:32], token[32:48])
...
```

每个 block 的 hash 包含了从序列开始到该 block 的所有 token，因此同一前缀的 block hash 完全相同。

**复用流程**：
1. 新请求到来，计算 prompt token 的 block hash
2. 在 hash 表中查找是否有已缓存的 block
3. 对于缓存命中的 block，直接复用其 KV Cache，跳过计算
4. 对于未命中的 block，正常计算并将结果存入缓存

### 2.2 Block 粒度的限制

vLLM 的前缀缓存以 block 为单位，通常为 16 tokens/block：

- **对齐要求**：复用只能在 block 边界处发生
- 如果共享前缀为 20 tokens，只有前 16 tokens（1 个 block）能被复用
- 不足一个 block 的部分必须重新计算

---

## 三、Radix Attention（SGLang）

Radix Attention 是 SGLang 提出的更精细的 KV Cache 管理方案，使用 Radix Tree（基数树）代替 Hash 表。

### 3.1 Radix Tree 结构

Radix Tree（压缩前缀树）的每条边对应一段 token 序列，根到某个节点的路径构成该节点对应的 token 序列前缀：

```
树结构示例（每个节点存储对应的 KV Cache）：

Root
├── "You are a helpful assistant. " (KV Cache A)
│   ├── "User: What is Python? " (KV Cache B)
│   │   └── "A: Python is a programming language..." (KV Cache C)
│   └── "User: What is Java? " (KV Cache D)
│       └── "A: Java is a compiled language..." (KV Cache E)
└── "You are a SQL expert. " (KV Cache F)
    └── "User: SELECT * " (KV Cache G)
```

**查找流程**：
1. 新请求 token 序列 = `["You are a helpful assistant. ", "User: What is C++? "]`
2. 从 Root 开始，找到最长匹配前缀 → 匹配到 KV Cache A
3. 只需计算 `"User: What is C++? "` 对应的 KV Cache

### 3.2 Radix Tree 的 LRU 淘汰

Radix Tree 的节点（KV Cache）会占用 GPU 显存，需要淘汰策略：

```
LRU（最近最少使用）淘汰：
- 每个节点记录最后被访问时间
- 当 GPU 显存不足时，优先淘汰最久未访问的节点
- 淘汰时从叶节点开始（不能淘汰仍被使用的节点）
```

### 3.3 Radix Attention vs vLLM APC 对比

| 特性 | vLLM APC | SGLang Radix Attention |
|------|---------|----------------------|
| 数据结构 | Hash Table（Block 级别） | Radix Tree（Token 级别） |
| 复用粒度 | Block（16 tokens）对齐 | 任意长度（Token 级别） |
| 是否自动 | 需要显式启用 | 默认开启，自动 |
| 多请求共享 | 支持 | 支持，且更精细 |
| 淘汰策略 | LRU Block 淘汰 | LRU Node 淘汰 |
| 树形结构复杂度 | O(1) 查找 | O(prefix_len) 查找，但树操作高效 |

---

## 四、前缀缓存的性能收益

### 4.1 Cache Hit Rate（缓存命中率）

缓存命中率 = 被复用的 KV Cache token 数 / 总 Prefill token 数

**典型场景的命中率**：

| 场景 | 典型命中率 | 性能提升 |
|------|---------|---------|
| 固定 system prompt（100 tokens） | 90%+ | TTFT 降低 50-80% |
| Few-shot QA（1000 token examples） | 70-90% | TTFT 降低 60-80% |
| 长文档 QA（5000 token 文档） | 80-95% | TTFT 降低 70-90% |
| 多轮对话 | 50-80% | TTFT 降低 30-60% |

### 4.2 减少 GPU 计算和内存

前缀缓存节省了：
- **Prefill 计算**：不需要重新计算 QKV 和 Attention
- **显存写入**：不需要重新写入 KV Cache（直接复用）

### 4.3 对 TTFT 的影响

```
不使用前缀缓存：
TTFT = Prefill(prompt_len) 时间

使用前缀缓存（命中率 80%）：
TTFT = Prefill(0.2 × prompt_len) 时间  → 降低约 80%
```

---

## 五、实际配置与使用

### 5.1 vLLM 开启 Prefix Caching

```python
from vllm import LLM

# 启用 Automatic Prefix Caching
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_prefix_caching=True,        # 开启 APC
    gpu_memory_utilization=0.9,         # 留更多空间给 KV Cache
    # max_model_len=8192,              # 可选：限制最大序列长度
)

# 多个请求共享相同前缀
shared_prefix = "You are a helpful assistant. " * 50  # 长 system prompt

responses = llm.generate([
    shared_prefix + "What is Python?",
    shared_prefix + "What is Java?",
    shared_prefix + "What is C++?",
])
# 第一个请求正常计算，后两个请求复用 KV Cache
```

### 5.2 SGLang 的前缀缓存（默认开启）

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, document, questions):
    s += sgl.system("You are a document QA assistant.")
    s += sgl.user(f"Document: {document}")  # 长文档前缀

    answers = []
    for question in questions:
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=100))
        answers.append(s["answer"])
    return answers

# SGLang 自动缓存 document 的 KV Cache
result = multi_turn_qa.run(
    document="...(长文档内容)...",
    questions=["问题1", "问题2", "问题3"],
)
```

---

## 六、前缀缓存的工程挑战

### 6.1 Prompt Template 一致性

前缀缓存要求请求的 token 序列完全一致才能命中。常见问题：

- 系统时间戳在 prompt 中（每次不同）→ 不能缓存
- 动态插入的内容（用户 ID、会话 ID）→ 放在 prompt 末尾，避免影响共享前缀
- BOS token 处理方式不一致 → 确保 tokenizer 配置一致

**最佳实践**：
```
设计 prompt 格式时，将固定部分放在最前面：
[System Prompt（固定）] + [Few-shot Examples（固定）] + [用户输入（变化）]

不要：
[用户名（变化）] + [System Prompt（固定）] + [用户输入（变化）]
```

### 6.2 缓存过期与一致性

KV Cache 与模型版本绑定，模型升级后需要清空缓存：
- vLLM 重启时自动清空
- 可以通过 API 手动清空缓存

### 6.3 多节点缓存（Distributed KV Cache）

在 PD 分离架构中，缓存可以存储在专门的 KV Cache 服务器上，所有 P 节点共享缓存（如 Mooncake 的设计）。

---

## 七、总结

前缀缓存是提升 LLM 推理性能的重要技术：

1. **核心价值**：避免重复计算相同前缀的 KV Cache，降低 TTFT
2. **vLLM 方案**：Hash-based Block 级别缓存（APC），需显式启用
3. **SGLang 方案**：Radix Tree，Token 级别精细缓存，默认开启
4. **最大化效益**：将固定内容（system prompt、few-shot）放在最前面
5. **适用场景**：API 服务、多轮对话、长文档 QA、代码补全等大量共享前缀的场景

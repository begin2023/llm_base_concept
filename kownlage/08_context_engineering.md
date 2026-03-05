# 8. Context Engineering（上下文工程）详解

---

## 一、概念定义

### 1.1 什么是 Context Engineering

Context Engineering（上下文工程）是指**系统性地设计、组织和优化输入给大语言模型的全部上下文信息**的工程实践。它不仅仅关注 prompt 本身的措辞，而是关注模型在推理时能够"看到"的所有信息的**全局最优组织**。

一个更准确的定义：**Context Engineering 是构建动态系统的学科，该系统在正确的时间，以正确的格式，提供正确的信息和工具给 LLM。**

这个概念由 Shopify CEO Tobi Lutke 等人在 2025 年初推广开来，强调了一个重要观点：**在 LLM 应用开发中，如何组织输入上下文的重要性远远超过了简单的提示词优化**。

### 1.2 上下文的组成

一个完整的 LLM 上下文包含以下部分：

```
┌─────────────────────────────────────────────────┐
│                   完整上下文                       │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  System Prompt（系统提示）                  │    │
│  │  - 角色定义、行为规则、输出格式要求           │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Tool Definitions（工具定义）               │    │
│  │  - 可用工具的名称、描述、参数 schema         │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Retrieved Context（检索到的上下文）         │    │
│  │  - RAG 检索的文档片段                       │    │
│  │  - 数据库查询结果                           │    │
│  │  - API 调用结果                             │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Conversation History（对话历史）           │    │
│  │  - 用户消息                                │    │
│  │  - 助手回复                                │    │
│  │  - 工具调用及其结果                         │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Memory（记忆）                             │    │
│  │  - 短期记忆（最近几轮对话）                  │    │
│  │  - 长期记忆（持久化的用户偏好、历史摘要）     │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Current User Input（当前用户输入）          │    │
│  │  - 最新的用户消息                           │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 1.3 与 Prompt Engineering 的区别

这是一个非常重要的概念区分：

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|-------------------|
| **范围** | 聚焦于提示词本身的措辞和结构 | 涵盖模型接收的全部输入的组织 |
| **关注点** | "如何写好一个提示" | "如何构建最优的信息输入系统" |
| **静态/动态** | 通常是静态的模板 | 通常是动态的、运行时决定的 |
| **包含内容** | 指令、示例、约束 | 检索结果、记忆、工具结果、上下文压缩 |
| **技术栈** | 文本编写技巧 | RAG、向量数据库、缓存、压缩算法 |
| **优化目标** | 单次回复质量 | 系统整体效果和效率 |
| **关系** | Context Engineering 的子集 | 更广泛的工程学科 |

**简单来说：Prompt Engineering 是 Context Engineering 的一部分。Context Engineering 还包括：信息检索策略、记忆管理、工具结果组织、上下文窗口分配、缓存策略等。**

---

## 二、核心技术

### 2.1 RAG（Retrieval-Augmented Generation，检索增强生成）

RAG 是 Context Engineering 中最重要的技术之一，核心思想是：**在模型生成回答之前，先从外部知识库中检索相关信息，将检索结果作为上下文提供给模型**。

#### 2.1.1 基础 RAG 架构

```
用户问题："量子退火算法的最新进展是什么？"
          ↓
┌──────────────────────────────────┐
│        Query Processing           │
│  问题分析 → 查询改写 → 查询扩展    │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────────────────┐
│        Retrieval（检索）           │
│                                  │
│  ┌──────────┐    ┌────────────┐  │
│  │ 向量检索   │    │ 关键词检索  │  │
│  │ (Dense)   │    │ (Sparse)   │  │
│  │ Embedding │    │ BM25/TF-IDF│  │
│  └─────┬────┘    └─────┬──────┘  │
│        └───────┬───────┘         │
│                ↓                 │
│          混合检索 + 重排序         │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────────────────┐
│      Context Assembly（上下文组装）│
│  - 选择 Top-K 最相关的文档片段     │
│  - 排序和去重                     │
│  - 添加来源引用                   │
│  - 拼接到 prompt 中               │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────────────────┐
│         LLM Generation           │
│  System: 基于以下参考资料回答...    │
│  Context: [检索到的文档片段]       │
│  User: 量子退火算法的最新进展...    │
│  → 生成回答                       │
└──────────────────────────────────┘
```

#### 2.1.2 高级 RAG 技术

**（1）查询改写（Query Rewriting）**

用户的原始问题可能表述模糊或不完整，查询改写通过 LLM 将原始查询优化为更适合检索的形式：

```
原始查询："那个谷歌做的量子的东西怎么样了"
改写后：
  - "Google Sycamore 量子处理器最新研究进展"
  - "Google 量子计算 2024 年成果"
  - "Google Willow 量子芯片"
```

**（2）HyDE（Hypothetical Document Embedding）**

先让 LLM 生成一个"假想答案"，然后用这个假想答案做向量检索（因为假想答案和真实文档在语义空间更接近）：

```
问题："量子纠错码的阈值是多少？"
LLM 生成假想答案："量子纠错码的错误阈值通常在 1% 左右，这意味着..."
用假想答案的 embedding 去检索 → 找到更精确的文档
```

**（3）多跳检索（Multi-hop Retrieval）**

复杂问题可能需要多次检索才能获得完整答案：

```
问题："Transformer论文的第一作者现在在哪家公司？"

第一跳检索："Transformer 论文 Attention is All You Need 作者"
  → 第一作者是 Ashish Vaswani

第二跳检索："Ashish Vaswani 2024 年 公司"
  → 他联合创办了 Essential AI
```

**（4）Agentic RAG**

将 RAG 与 Agent 结合，Agent 自主决定何时检索、检索什么、是否需要多次检索：

```python
class AgenticRAG:
    def answer(self, question):
        # Agent 分析问题，决定检索策略
        plan = self.planner.plan(question)

        context = []
        for step in plan.steps:
            if step.type == "vector_search":
                results = self.vector_db.search(step.query, top_k=step.k)
                context.extend(results)
            elif step.type == "sql_query":
                results = self.db.execute(step.sql)
                context.append(results)
            elif step.type == "web_search":
                results = self.web.search(step.query)
                context.extend(results)

        # 基于所有收集到的上下文生成回答
        return self.llm.generate(question=question, context=context)
```

#### 2.1.3 RAG 与推理引擎的关系

RAG 对推理引擎有直接的影响：

1. **输入长度增加**：RAG 注入的上下文通常占用数千到数万 token，增加了 prefill 阶段的计算量
2. **Prefix Caching 价值**：如果多个用户的问题检索到相同的文档，prefix cache 可以避免重复计算
3. **KV Cache 压力**：更长的上下文意味着更大的 KV Cache 内存占用
4. **延迟影响**：TTFT 与输入长度近似成线性关系，RAG 显著增加了 TTFT

### 2.2 长上下文管理

#### 2.2.1 挑战

现代模型支持越来越长的上下文窗口（128K、200K、甚至 1M+ token），但长上下文带来的挑战：

- **计算成本**：Self-Attention 的计算复杂度为 O(n^2)（标准 Attention），即使有 Flash Attention 优化，长上下文的 prefill 仍然很慢
- **"迷失在中间"（Lost in the Middle）**：研究表明，模型对上下文中间部分的信息注意力较弱，重要信息放在开头或结尾效果更好
- **内存占用**：KV Cache 随上下文长度线性增长，128K 上下文的 KV Cache 可能占用数十 GB 显存
- **质量下降**：上下文过长时，模型可能被不相关信息干扰，回答质量反而下降

#### 2.2.2 长上下文管理策略

**策略一：信息排序优化**

```
┌──────────────────────────────────┐
│          最优信息排列              │
│                                  │
│  ████████  开头：最重要的信息     │
│  ░░░░░░░░  中间：次要的信息       │
│  ████████  结尾：重要的总结/指令  │
│                                  │
│  原因："Lost in the Middle" 现象   │
│  模型对开头和结尾的注意力最强      │
└──────────────────────────────────┘
```

**策略二：分层上下文**

将上下文分为不同层次，按需加载：

```
Level 1 - 核心上下文（始终包含）:
  - System Prompt
  - 当前用户消息
  - 最重要的工具定义

Level 2 - 会话上下文（最近 N 轮）:
  - 最近的对话历史
  - 最近的工具调用结果

Level 3 - 检索上下文（动态加载）:
  - RAG 检索结果
  - 相关的长期记忆

Level 4 - 背景上下文（按需加载）:
  - 完整的文档
  - 历史对话摘要
```

**策略三：滑动窗口 + 摘要**

```
对话轮次 1-10 → LLM 总结 → 摘要（~200 token）
对话轮次 11-20 → LLM 总结 → 摘要（~200 token）
对话轮次 21-25 → 完整保留 （~2000 token）

上下文 = [System] + [摘要1] + [摘要2] + [最近5轮完整对话] + [当前消息]
```

### 2.3 上下文压缩

#### 2.3.1 为什么需要压缩

上下文窗口是宝贵的"不动产"，每一个 token 都有成本（计算时间、内存、API 费用）。上下文压缩的目标是在不丢失关键信息的前提下，减少上下文的 token 数量。

#### 2.3.2 压缩技术

**（1）LLM-based 压缩**

使用 LLM 本身来压缩上下文：

```
原始文档（3000 token）:
"在2024年1月15日，OpenAI 发布了...（大量细节）...总结来说，GPT-5 在推理能力上
 有了显著提升，主要表现在数学推理（提升40%）和代码生成（提升35%）方面。"

压缩指令："请保留关键事实，压缩到 300 token 以内"

压缩后（250 token）:
"OpenAI 2024年1月发布 GPT-5，推理能力显著提升：数学推理+40%，代码生成+35%。"
```

**（2）Selective Context（选择性上下文）**

基于信息量（self-information/entropy）评估每个 token 或 sentence 的重要性，保留信息量高的部分：

```python
def selective_context(text, target_ratio=0.5):
    sentences = split_sentences(text)
    # 计算每个句子的自信息量
    scores = [compute_self_information(s) for s in sentences]
    # 保留信息量最高的句子
    threshold = np.percentile(scores, (1 - target_ratio) * 100)
    selected = [s for s, score in zip(sentences, scores) if score >= threshold]
    return " ".join(selected)
```

**（3）LLMLingua / LongLLMLingua**

微软提出的上下文压缩方法，使用小模型（如 GPT-2）评估 token 的困惑度，删除低困惑度（可预测的、信息量低的）token：

```
原文："The weather in Beijing today is very nice and sunny with clear blue skies"
压缩："weather Beijing today nice sunny clear blue skies"
```

压缩比可达 2x-20x，但需要注意压缩过多可能导致信息丢失。

**（4）Embedding-based 压缩**

将长文本编码为固定长度的向量，作为"软 token"注入模型。这种方法在学术研究中较多，实际生产中使用较少。

#### 2.3.3 压缩与推理性能

上下文压缩对推理性能的影响：

```
假设原始上下文 10000 token，压缩到 3000 token：

Prefill 阶段：
  - 原始：计算 10000 token 的注意力 → ~500ms
  - 压缩后：计算 3000 token 的注意力 → ~150ms
  - TTFT 降低约 70%

KV Cache：
  - 原始：10000 × 2 × num_layers × hidden_dim × sizeof(float16)
  - 压缩后：3000 × 2 × num_layers × hidden_dim × sizeof(float16)
  - 显存占用减少约 70%

吞吐量：
  - KV Cache 减少 → 同等显存下可以服务更多并发请求
```

### 2.4 上下文缓存（Prefix Caching / Context Caching）

#### 2.4.1 概念

Prefix Caching 是推理引擎中的重要优化技术，核心思想是：**如果多个请求的上下文有相同的前缀，可以缓存这个前缀的 KV Cache，避免重复计算**。

这在以下场景中非常有价值：
- **多轮对话**：每一轮的上下文都包含之前所有轮次的内容（相同前缀）
- **RAG**：多个用户可能检索到相同的文档片段
- **共享 System Prompt**：所有请求共享相同的 System Prompt
- **批量处理**：对同一文档的多个问题共享文档上下文

#### 2.4.2 工作原理

```
请求 1 的上下文：[System Prompt | 文档A | 用户问题1]
                 ^^^^^^^^^^^^^^^^^^^^^^^^
                    公共前缀（可缓存）

请求 2 的上下文：[System Prompt | 文档A | 用户问题2]
                 ^^^^^^^^^^^^^^^^^^^^^^^^
                    命中缓存，直接复用 KV Cache

请求 3 的上下文：[System Prompt | 文档B | 用户问题3]
                 ^^^^^^^^^^^^^^^
                   部分命中缓存
```

#### 2.4.3 推理引擎中的实现

**vLLM 的 Automatic Prefix Caching (APC)**

vLLM 的实现基于 **token block** 的哈希匹配：

```
1. 将 token 序列按固定块大小（如 16 token/block）分块
2. 对每个块计算哈希值（包含前面所有块的哈希，形成哈希链）
3. 新请求到来时，逐块检查哈希是否命中缓存
4. 命中的块直接复用 KV Cache，未命中的块重新计算

Block Hash 计算：
  Block 0: hash(token[0:16])
  Block 1: hash(Block0_hash, token[16:32])
  Block 2: hash(Block1_hash, token[32:48])
  ...
```

启用方式：
```bash
vllm serve model_name --enable-prefix-caching
```

**SGLang 的 RadixAttention**

SGLang 使用 **Radix Tree（基数树）** 来高效管理前缀缓存：

```
              Root
             /    \
     [System Prompt]  [Other Prefix]
         /       \
  [Doc A]       [Doc B]
   /    \         |
[Q1]  [Q2]     [Q3]
```

RadixAttention 的优势：
- 支持任意前缀的匹配（不限于公共前缀）
- 自动 LRU 淘汰策略
- 对多轮对话和分支对话特别友好

```python
# SGLang 的使用示例（自动启用 prefix caching）
import sglang as sgl

@sgl.function
def multi_turn(s, doc, questions):
    s += sgl.system("你是一个有用的助手。")
    s += sgl.user(f"阅读以下文档：\n{doc}")  # 公共前缀

    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=200))
    # 第二个问题开始自动复用前面的 KV Cache
```

**Google Gemini API 的 Context Caching**

Google 在 Gemini API 中提供了显式的上下文缓存功能：

```python
import google.generativeai as genai

# 创建缓存（显式指定要缓存的上下文）
cache = genai.caching.CachedContent.create(
    model="gemini-1.5-pro",
    display_name="my_doc_cache",
    system_instruction="你是一个文档分析专家。",
    contents=[large_document],  # 要缓存的大型文档
    ttl=datetime.timedelta(hours=1)  # 缓存有效期
)

# 使用缓存进行多次查询（只计费增量部分）
model = genai.GenerativeModel.from_cached_content(cache)
response1 = model.generate_content("文档的主要观点是什么？")
response2 = model.generate_content("文档中提到了哪些数据？")
```

**Anthropic Claude API 的 Prompt Caching**

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": very_long_system_prompt,  # 长的 system prompt
            "cache_control": {"type": "ephemeral"}  # 标记为可缓存
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": long_document,
                    "cache_control": {"type": "ephemeral"}  # 标记为可缓存
                },
                {
                    "type": "text",
                    "text": "请总结这篇文档。"
                }
            ]
        }
    ]
)
```

#### 2.4.4 Prefix Caching 的性能影响

```
场景：10000 token 的公共前缀 + 100 token 的用户问题

无缓存：
  Prefill 10100 token → ~500ms TTFT

有缓存（命中）：
  复用 10000 token 的 KV Cache + Prefill 100 token → ~10ms TTFT
  加速比：~50x

成本影响（API 场景）：
  - Anthropic: 缓存命中部分按 1/10 价格计费
  - Google: 缓存命中部分按 1/4 价格计费
  - 自部署: 减少 GPU 计算时间
```

### 2.5 上下文窗口管理策略

#### 2.5.1 Token Budget 分配

在有限的上下文窗口中，如何分配 token 预算是一个关键决策：

```
假设模型上下文窗口为 128K token，典型分配：

┌─────────────────────────────────────────────┐
│  System Prompt:           2,000 tokens (2%) │
│  Tool Definitions:        3,000 tokens (2%) │
│  RAG Retrieved Context:  50,000 tokens (39%)│
│  Conversation History:   30,000 tokens (23%)│
│  Long-term Memory:       10,000 tokens (8%) │
│  Current User Input:      1,000 tokens (1%) │
│  Reserved for Output:    32,000 tokens (25%)│
│                                             │
│  Total:                 128,000 tokens      │
└─────────────────────────────────────────────┘
```

**分配原则**：
- 始终为模型输出预留足够空间（通常至少预留总窗口的 20-25%）
- System Prompt 应该尽量精简
- RAG 上下文的量取决于任务复杂度
- 对话历史采用滑动窗口 + 摘要策略

#### 2.5.2 动态上下文管理

```python
class ContextManager:
    def __init__(self, max_tokens=128000, reserved_output=32000):
        self.max_input_tokens = max_tokens - reserved_output
        self.priorities = {
            "system_prompt": 1,      # 最高优先级，不可删除
            "current_input": 2,      # 不可删除
            "tool_definitions": 3,   # 尽量保留
            "recent_history": 4,     # 保留最近几轮
            "rag_context": 5,        # 可以截断
            "old_history": 6,        # 可以摘要或删除
            "memory": 7,             # 可以按相关性筛选
        }

    def build_context(self, components):
        """按优先级构建上下文，确保不超出窗口限制"""
        context = []
        remaining_tokens = self.max_input_tokens

        # 按优先级排序
        sorted_components = sorted(components,
                                   key=lambda x: self.priorities[x.type])

        for component in sorted_components:
            tokens = count_tokens(component.content)
            if tokens <= remaining_tokens:
                context.append(component)
                remaining_tokens -= tokens
            else:
                # 尝试截断或压缩
                truncated = self.truncate_or_compress(
                    component, remaining_tokens)
                if truncated:
                    context.append(truncated)
                    remaining_tokens -= count_tokens(truncated.content)

        return context

    def truncate_or_compress(self, component, max_tokens):
        """根据组件类型选择截断或压缩策略"""
        if component.type == "rag_context":
            # RAG 结果：保留相关性最高的片段
            return self.select_top_chunks(component, max_tokens)
        elif component.type == "old_history":
            # 旧对话：生成摘要
            return self.summarize_history(component, max_tokens)
        elif component.type == "memory":
            # 记忆：按相关性筛选
            return self.filter_by_relevance(component, max_tokens)
        return None
```

#### 2.5.3 上下文窗口与推理性能的关系

上下文长度对推理引擎各指标的影响：

```
TTFT（首 Token 延迟）：
  TTFT ∝ input_length（近似线性，受 Flash Attention 优化影响）

  1K tokens  → ~50ms
  10K tokens → ~200ms
  50K tokens → ~800ms
  128K tokens → ~3000ms

KV Cache 内存占用：
  KV Cache Size = 2 × num_layers × num_heads × head_dim × seq_len × dtype_size

  对于 Llama-3-70B（80层，64头，128维，FP16）：
    1K tokens  → ~0.6 GB
    10K tokens → ~6 GB
    128K tokens → ~80 GB  （单个请求！）

吞吐量影响：
  - 长上下文占用更多 KV Cache → 同等显存能处理的并发数下降
  - Prefill 计算量增加 → Prefill 阶段可能成为瓶颈
  - 在 PD 分离架构中，长上下文请求应路由到 Prefill 节点

Decode 阶段每步延迟：
  - 每步 Attention 需要访问全部 KV Cache
  - seq_len 越长，每步的内存带宽需求越大
  - 但由于 Decode 是 memory-bound，影响是亚线性的
```

---

## 三、上下文工程最佳实践

### 3.1 信息密度最大化

**原则：每一个 token 都应该有价值。**

```
差的做法（信息密度低）：
  "接下来我会给你提供一些关于这个主题的背景信息，请你仔细阅读这些信息，
   然后基于这些信息来回答我的问题。以下是背景信息的详细内容："

好的做法（信息密度高）：
  "基于以下参考资料回答问题：\n\n---\n{document}\n---"
```

### 3.2 结构化组织

使用清晰的结构标记组织上下文：

```xml
<system_context>
你是一个金融分析助手，专注于A股市场分析。
</system_context>

<reference_documents>
<doc id="1" source="2024年年报" relevance="0.95">
  公司2024年营收同比增长25%...
</doc>
<doc id="2" source="行业报告" relevance="0.87">
  半导体行业整体景气度回升...
</doc>
</reference_documents>

<conversation_history>
User: 分析一下这家公司的财务状况
Assistant: [之前的分析内容]
</conversation_history>

<current_query>
这家公司的估值是否合理？请结合行业对比分析。
</current_query>
```

### 3.3 动态上下文选择

根据用户当前问题，动态选择最相关的上下文：

```python
def select_context(query, available_context):
    """根据查询动态选择上下文"""

    # 1. 语义相似度筛选
    query_embedding = embed(query)
    relevance_scores = [
        cosine_similarity(query_embedding, embed(ctx))
        for ctx in available_context
    ]

    # 2. 按相关性排序，选择 Top-K
    sorted_ctx = sorted(zip(available_context, relevance_scores),
                        key=lambda x: x[1], reverse=True)

    # 3. 填充上下文直到预算用完
    selected = []
    budget = MAX_CONTEXT_TOKENS
    for ctx, score in sorted_ctx:
        if score < MIN_RELEVANCE_THRESHOLD:
            break
        tokens = count_tokens(ctx)
        if tokens <= budget:
            selected.append(ctx)
            budget -= tokens

    return selected
```

### 3.4 上下文质量评估

建立上下文质量的评估体系：

```
评估维度：
1. 相关性（Relevance）：上下文信息与用户问题的相关度
2. 完整性（Completeness）：上下文是否包含回答问题所需的全部信息
3. 冗余度（Redundancy）：上下文中是否有重复或冗余信息
4. 新鲜度（Freshness）：信息是否是最新的
5. 噪声（Noise）：上下文中是否混入了不相关的干扰信息
6. 格式（Format）：上下文的组织方式是否便于模型理解

量化指标：
- Precision@K: 检索的 Top-K 文档中相关文档的比例
- Recall: 所有相关文档中被检索到的比例
- Context Utilization: 模型实际使用了多少上下文信息
- Faithfulness: 模型回答是否忠于上下文（而非幻觉）
```

---

## 四、前沿发展

### 4.1 Infinite Context（无限上下文）

- **Google Infini-Attention**：将注意力机制与压缩记忆结合，理论上支持无限长上下文
- **StreamingLLM**：通过保留 Attention Sink（注意力汇聚 token）实现无限长流式推理
- **Ring Attention**：跨设备分布式处理超长上下文

### 4.2 Context Distillation（上下文蒸馏）

将长上下文的信息"蒸馏"到模型参数中，使模型无需显式上下文即可回忆相关信息。

### 4.3 Adaptive Context（自适应上下文）

模型根据任务复杂度自动调节所需的上下文量，简单问题用少量上下文，复杂问题动态扩展上下文。

### 4.4 Multi-modal Context（多模态上下文）

上下文不再局限于文本，还包括图像、音频、视频、表格等多种模态的信息。如何在有限的窗口中高效组织多模态上下文是新的挑战。

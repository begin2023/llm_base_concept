# 25. 推理续写（Inference Continuation）

---

## 一、什么是推理续写

推理续写（Inference Continuation），也叫**多轮推理链续写**，是指 LLM 在推理（Reasoning）过程中，能够基于前面已有的思考链（Chain of Thought, CoT）继续生成更深入的推理步骤，而不是从头重新推理。

更广义的"续写"还包括：

1. **对话续写**：多轮对话中，基于历史上下文继续生成
2. **长文续写**：给定文章开头，继续生成剩余内容
3. **代码续写**：给定代码上下文，继续补全代码
4. **推理链续写**：给定部分思考过程（CoT），继续推理

在工程上，所有这些"续写"本质上都是：**给定一个前缀（prefix），让模型从这个前缀后面继续生成 token**。

---

## 二、推理续写的工程实现

### 2.1 基本原理

LLM 的 next-token prediction 天然支持续写：

```
前缀：[token_1, token_2, ..., token_k]
→ Prefill：计算 k 个 token 的 KV Cache
→ Decode：从 token_{k+1} 开始自回归生成
```

关键在于：**前缀的 KV Cache 可以被缓存和复用**。

### 2.2 在 vLLM 中实现续写

```python
from vllm import LLM, SamplingParams

llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# 第一次：生成部分推理链
partial_cot_prompt = """<|im_start|>user
解方程 2x² + 5x - 3 = 0
<|im_end|>
<|im_start|>assistant
<think>
使用求根公式：x = (-b ± √(b²-4ac)) / 2a
其中 a=2, b=5, c=-3
"""

# 生成前 200 个 token 的推理
params = SamplingParams(max_tokens=200)
output = llm.generate(partial_cot_prompt, params)

# 续写：基于已有的推理链继续
continuation_prompt = partial_cot_prompt + output.outputs[0].text + "\n继续推导..."
params2 = SamplingParams(max_tokens=500)
full_output = llm.generate(continuation_prompt, params2)
```

### 2.3 前缀缓存加速续写（Prefix Caching）

当续写的前缀较长时，Prefix Caching 可以避免重复计算 KV Cache：

```python
# 启用 Prefix Caching
llm = LLM(
    model="...",
    enable_prefix_caching=True,  # vLLM
)

# 首次请求：计算并缓存前缀的 KV Cache
first_response = llm.generate(long_prefix + "第一个问题", params)

# 续写请求：前缀的 KV Cache 已缓存，只计算新增 token
second_response = llm.generate(long_prefix + "第二个问题", params)
# → 只需计算 "第二个问题" 的 KV Cache，long_prefix 部分直接复用
```

---

## 三、思维链续写（CoT Continuation）

### 3.1 背景：Thinking Mode 与推理模型

以 DeepSeek-R1、OpenAI o1 为代表的推理模型，会生成很长的"内部思考链"（`<think>...</think>`），然后再给出答案。

这些模型的推理过程可能包含：
- 分析问题
- 尝试不同方法
- 自我纠错
- 逐步推导

### 3.2 推理续写的价值

**场景 1：截断续写（Truncated Continuation）**

当模型生成 CoT 时突然中断（网络超时、token 限制等），可以从中断处续写，而不是重新生成。

**场景 2：引导推理（Guided Reasoning）**

通过插入中间步骤，引导模型按照特定路径推理：

```
原始 prompt → 生成部分 CoT → 人工插入修正/引导 → 续写完成推理
```

**场景 3：分段推理（Segmented Reasoning）**

对于超长推理任务，分段生成和验证推理链，避免 context window 溢出。

### 3.3 示例：数学推理续写

```python
# 初始 prompt
problem = "证明：对任意正整数 n，n³ + 2n 能被 3 整除"

# 模型生成部分推理
partial_solution = """
<think>
我需要证明 n³ + 2n = n(n² + 2) 能被 3 整除。

考虑 n 除以 3 的余数，有三种情况：
情况 1：n = 3k（n 是 3 的倍数）
  n³ + 2n = (3k)³ + 2(3k) = 27k³ + 6k = 3(9k³ + 2k) ✓

情况 2：n = 3k + 1
  n³ + 2n = (3k+1)³ + 2(3k+1)
          = 27k³ + 27k² + 9k + 1 + 6k + 2
          = 27k³ + 27k² + 15k + 3
"""

# 续写（情况 3 还没完成）
continuation_prompt = problem + partial_solution + "          = 3(9k³ + 9k² + 5k + 1) ✓\n情况 3："
```

---

## 四、长上下文推理中的续写挑战

### 4.1 Context Window 限制

推理模型可能会生成非常长的 CoT（DeepSeek-R1 有时会生成 10000+ tokens）。当 CoT 接近 context window 上限时，需要特殊处理：

**方案 1：截断旧内容**
- 删除最早的部分 CoT
- 风险：丢失早期重要推理步骤

**方案 2：CoT 压缩（Summarization）**
- 对已完成的推理步骤进行摘要，压缩 KV Cache 占用
- 保留关键信息，减少 token 数量

**方案 3：分段推理（Chunked Reasoning）**
- 将大问题分解为子问题
- 每个子问题独立推理，结果汇总

### 4.2 KV Cache 管理

长推理链的 KV Cache 占用很大：

```
推理 token 数：10000
KV Cache per token（Llama-3-8B BF16）：
= 2 × 32层 × 8heads × 128dim × 2bytes = 131 KB/token
总 KV Cache = 10000 × 131 KB = 1.31 GB（仅一个请求）
```

这要求推理系统有高效的 KV Cache 管理（PagedAttention / RadixAttention）。

---

## 五、工程实践：续写 API 设计

### 5.1 OpenAI Compatible API 续写

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

# 使用 suffix 参数（某些实现支持）
completion = client.completions.create(
    model="gpt-4",
    prompt="原有的前缀内容...",
    max_tokens=500,
)

# 流式续写
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "已有的部分回答..."}  # 续写这里
    ],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 5.2 注入部分 Assistant 回复

通过在消息列表中添加部分 assistant 回复（`content` 非空），让模型从这个位置续写：

```python
messages = [
    {"role": "system", "content": "你是一个数学专家"},
    {"role": "user", "content": "计算 1+1"},
    {"role": "assistant", "content": "让我来计算：\n1 + 1 = "}  # 模型从这里续写
]
```

这种技术叫做 **"Assistant Pre-fill"**（预填充 Assistant 回复），是控制模型输出格式的重要技术。

### 5.3 强制续写格式

```python
# 强制模型以 JSON 格式输出（通过续写开头）
messages = [
    {"role": "user", "content": "提取实体：今天北京天气晴，气温25度"},
    {"role": "assistant", "content": '{"entities": ['}  # 预填充 JSON 开头
]
# 模型会续写 {"entities": [ 后面的内容，确保输出是合法 JSON
```

---

## 六、推理续写与 Speculative Decoding 的结合

推理续写可以与投机解码结合：

```
前缀 KV Cache（已缓存）
    ↓
草稿模型：快速生成 K 个候选推理 token
    ↓
目标模型：验证候选 token（一次前向传播）
    ↓
接受/拒绝，继续生成
```

由于推理链的 token 通常高度可预测（逐步推导，风格一致），投机解码的接受率较高，加速效果好。

---

## 七、推理中断与恢复

### 7.1 断点续推（Checkpoint Resume）

对于超长推理任务：

```python
# 保存中间状态
checkpoint = {
    "prompt": original_prompt,
    "generated_tokens": current_output_tokens,
    "timestamp": datetime.now(),
}
save_checkpoint(checkpoint)

# 从断点恢复
checkpoint = load_checkpoint()
resume_prompt = checkpoint["prompt"] + detokenize(checkpoint["generated_tokens"])
output = llm.generate(resume_prompt, SamplingParams(max_tokens=remaining_tokens))
```

### 7.2 在分布式系统中的续写

在多节点系统（PD 分离）中，续写意味着：
- P 节点：对续写的前缀做 Prefill（但前缀 KV Cache 可能已在 D 节点缓存）
- D 节点：如果前缀 KV Cache 仍有效，直接续写 Decode
- KV Cache 的跨节点传输和缓存管理

---

## 八、总结

推理续写的核心要点：

1. **本质**：给定前缀，从前缀后继续自回归生成
2. **工程优化**：Prefix Caching 避免重复计算前缀 KV Cache
3. **关键技术**：Assistant Pre-fill、流式输出、断点恢复
4. **推理模型的特殊性**：长 CoT 需要高效 KV Cache 管理
5. **应用场景**：多轮对话、代码补全、数学推理、长文生成

对于 DeepSeek-R1 等推理模型，推理续写技术尤为重要，因为其 CoT 可能极长，如何高效管理和续写长推理链是工程实践的核心挑战。

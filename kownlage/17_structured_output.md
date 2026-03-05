# 17. 结构化输出（Structured Output）

---

## 一、什么是结构化输出

结构化输出（Structured Output）是指让 LLM 生成符合特定格式约束的文本，而不是自由文本。最常见的形式包括：

- **JSON 格式**：生成符合特定 JSON schema 的输出
- **正则表达式约束**：输出匹配特定正则模式的字符串
- **枚举约束**：输出只能是预定义选项中的一个
- **CFG 约束**：符合上下文无关文法（Context-Free Grammar）的输出
- **函数调用（Function Call）格式**：生成符合工具调用格式的 JSON

结构化输出在工程应用中极为重要，因为下游系统通常需要解析 LLM 的输出，自由文本难以可靠解析，而结构化输出保证了输出的可程序化处理性。

---

## 二、核心实现原理：引导解码（Guided Decoding）

结构化输出的核心技术是**引导解码（Guided Decoding）**，也称为**受限解码（Constrained Decoding）**。

### 2.1 基本原理

LLM 在每个 decode step 生成 logits（每个 token 的原始分数），通过 softmax 转换为概率分布，然后从中采样。

引导解码的做法是：在采样之前，将不合法的 token 的 logit 设为 `-inf`（或极小值），使其采样概率为 0。这样模型只能从合法的 token 中采样，从而保证输出符合约束。

```
原始 logits: [0.5, 0.3, 0.8, 0.1, ...]  (vocab_size 个值)
                                          ↓
mask（0/1 或 -inf/0）: 根据当前约束状态，计算哪些 token 合法
                                          ↓
masked logits: [0.5, -inf, 0.8, -inf, ...]
                                          ↓
采样：只能从合法 token 中采样
```

### 2.2 约束状态机（Constraint State Machine）

为了高效地判断"当前哪些 token 合法"，通常将约束转换为一个有限状态机（FSM）或类似结构：

- **当前状态**：记录已生成的输出在 FSM 中的位置
- **合法 token 集合**：在当前状态下，能使 FSM 状态合法转移的 token
- **状态转移**：每生成一个 token，FSM 的状态相应更新

对于 JSON schema，常用的方法是将 schema 先编译为正则表达式，再将正则表达式编译为 FSM。

---

## 三、JSON Schema 约束的实现

### 3.1 JSON Schema → 正则表达式 → FSM

以要求输出 `{"name": "string", "age": integer}` 为例：

1. **解析 JSON Schema**：确定各字段的类型和约束
2. **生成正则表达式**：将 schema 转换为能匹配所有合法 JSON 的正则表达式
3. **编译为 FSM**：将正则表达式编译为 DFA（确定性有限自动机）
4. **推理时执行**：每个 decode step 查询当前 FSM 状态，获取合法 token 集合

### 3.2 Token 级别的挑战

LLM 使用的是 BPE/SentencePiece tokenizer，一个 token 可能对应多个字符。这引入了复杂性：

- 一个 token 可能跨越 FSM 的多个状态转移
- 需要预计算每个 token 对应的字符序列，判断该 token 是否会导致 FSM 进入非法状态

常见解决方案：将 FSM 与 tokenizer 词表做预处理，建立"当前 FSM 状态 → 合法 token 集合"的映射表（称为 token mask cache）。

### 3.3 Outlines 库

[Outlines](https://github.com/outlines-dev/outlines) 是目前最流行的结构化输出实现库：

- 将约束编译为 FSM，并预处理 tokenizer 词表
- 缓存 FSM 状态到 token mask 的映射，避免推理时重复计算
- 支持 JSON schema、正则表达式、CFG 等多种约束类型
- vLLM 和 SGLang 都集成了 Outlines 的核心逻辑

---

## 四、各框架的实现

### 4.1 vLLM 中的结构化输出

vLLM 通过 `guided_decoding` 参数支持结构化输出：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(
    guided_decoding={
        "json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
)
output = llm.generate("Tell me about Alice", sampling_params)
```

vLLM 的实现细节：
- 在 `SamplingMetadata` 中携带约束信息
- 在每个 decode step 的 logits 处理阶段应用 mask
- 支持 batch 中不同请求有不同的约束（每个请求独立维护 FSM 状态）

### 4.2 SGLang 中的结构化输出

SGLang 在结构化输出上有独特优势：
- **RegexGuide**：基于正则表达式的引导解码
- **XGrammar**：更高效的 CFG 约束解码引擎
- SGLang 的 Structured Generation 是原生设计，与框架深度集成
- 支持在 Continuous Batching 中每个请求有不同的结构化约束

### 4.3 OpenAI API 的 Structured Output

OpenAI 在 2024 年推出了官方的 Structured Output 支持：
```python
from openai import OpenAI
from pydantic import BaseModel

class Response(BaseModel):
    name: str
    age: int

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=Response,
)
```

---

## 五、XGrammar：更高效的约束解码

XGrammar 是 SGLang 团队开发的高效结构化输出引擎：

### 5.1 核心思路

- 将 JSON schema 或 CFG 编译为 Pushdown Automaton（PDA，下推自动机）
- 对 PDA 做"上下文无关预处理"，将很多 token mask 计算提前到编译期
- 大幅减少推理时的 mask 计算开销

### 5.2 性能优势

- 相比 Outlines，XGrammar 在复杂 schema 下速度快 10-100 倍
- 支持增量式 mask 更新，只在 FSM 状态改变时重新计算
- GPU 加速的 mask apply 操作

---

## 六、性能开销与优化

### 6.1 主要开销来源

1. **Mask 计算**：每个 decode step 需要计算当前合法的 token 集合
2. **FSM 状态更新**：维护每个请求的约束状态
3. **Batch 内异构约束**：不同请求有不同约束时，难以向量化

### 6.2 优化手段

- **预编译和缓存**：将 schema → FSM → token mask 的计算在推理开始前完成
- **惰性计算**：只在 FSM 状态发生变化时才重新计算 mask
- **GPU 上的 mask 应用**：mask 的应用（设置 -inf）是向量操作，可以高效地在 GPU 上执行
- **公共约束复用**：如果多个请求共享同一 schema，可以共享 FSM 和 mask 计算

### 6.3 典型性能数据

- 对于简单 schema（少量字段）：额外开销 < 5%
- 对于复杂 schema（嵌套、数组等）：额外开销 5-20%
- 使用 XGrammar 等优化工具可将开销降至接近 0

---

## 七、应用场景

| 场景 | 约束类型 | 示例 |
|------|---------|------|
| 信息抽取 | JSON Schema | `{"entity": "string", "type": "string"}` |
| 分类任务 | 枚举约束 | `positive \| negative \| neutral` |
| 代码生成 | 正则/CFG | 合法的 Python 函数签名 |
| Function Call | JSON Schema | 工具参数的 JSON 格式 |
| 表单填写 | JSON Schema | 特定字段类型和格式 |
| SQL 生成 | CFG | 合法的 SQL 语法 |

---

## 八、总结

结构化输出是 LLM 工程落地的关键技术：

1. **核心机制**：引导解码 = 在采样前对不合法 token 施加 `-inf` mask
2. **约束表达**：FSM（有限状态机）是表达约束的通用手段
3. **主流实现**：Outlines（通用）、XGrammar（高性能）
4. **框架支持**：vLLM、SGLang 均已原生集成
5. **零幻觉保证**：结构化输出从数学上保证输出合法，无需后处理解析修复

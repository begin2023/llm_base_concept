# 34. Guided Decoding / Constrained Decoding

---

## 一、引导解码与受限解码概述

引导解码（Guided Decoding）和受限解码（Constrained Decoding）是两个密切相关的概念，都指在 LLM 生成过程中施加约束，确保输出满足特定格式或规则。

**两者的区别**：
- **Constrained Decoding（受限解码）**：更严格，从数学上保证输出 100% 满足约束（不可能违反）
- **Guided Decoding（引导解码）**：更广义，包括软引导（通过调整 logits 倾向于满足约束，但不保证）

在实践中两者常互换使用，本章以严格的约束解码为主。

---

## 二、核心技术：Logit Masking（Logit 遮蔽）

约束解码的基本机制：在每个 decode step，将不合法的 token 的 logit 设为 $-\infty$，使其采样概率为 0。

```
合法 token 集合: S_valid
非合法 token:   vocab \ S_valid

logits_masked[token] = logits[token]  if token ∈ S_valid
logits_masked[token] = -inf            if token ∉ S_valid

probs = softmax(logits_masked)
next_token = sample(probs)  # 只能从 S_valid 中采样
```

关键问题：**如何高效计算每个 decode step 的合法 token 集合？**

---

## 三、有限状态机（FSM）方法

### 3.1 FSM 基础

将约束表达为有限状态机（Finite State Machine）：

- **状态（State）**：当前生成进度
- **转移（Transition）**：接收一个字符，进入下一个状态
- **合法 token 集合**：在当前状态下，接收该 token 的字符后，FSM 不会进入"死状态"（即不会违反约束）

**示例（JSON 对象约束）**：
```
State 0: 期望 '{'
State 1: 期望 '"'（开始一个 key）或 '}'（空对象）
State 2: 期望 key 字符串内容
State 3: 期望 '"' 结束 key
State 4: 期望 ':'
State 5: 期望 value（任意类型）
...
```

### 3.2 从约束到 FSM

**正则表达式 → DFA（确定性有限自动机）**：
- 正则表达式可以转换为 NFA（非确定性有限自动机）
- NFA 可以转换为等价的 DFA
- 工具：Python 的 `regex`、`interegular` 等库

**JSON Schema → 正则表达式 → DFA**：
- 将 JSON Schema 转换为等价的正则表达式
- 再转为 DFA
- 工具：Outlines 库实现了这一完整流程

### 3.3 Token 级别的处理

由于 LLM 的词表中每个 token 可能对应多个字符，需要处理 token 与 FSM 状态的对应关系：

```python
# 预处理：对词表中每个 token，计算它在每个 FSM 状态下是否合法
token_mask_cache = {}  # {(fms_state, token_id): bool}

for state in all_fms_states:
    valid_tokens = []
    for token_id, token_str in enumerate(vocab):
        # 模拟 FSM 处理 token_str
        next_states = fms.process_string(state, token_str)
        if next_states is not None:  # 不会进入死状态
            valid_tokens.append(token_id)
    token_mask_cache[state] = valid_tokens
```

**挑战**：一个 token 可能跨越 FSM 的多个状态。处理方案：
- Token 的字符序列必须使 FSM 在处理完这些字符后处于有效状态
- 即使中间经过了多个状态，最终状态必须是非死状态

---

## 四、XGrammar：高效约束解码引擎

XGrammar（Chen et al. 2024）是 SGLang 团队开发的高性能约束解码引擎。

### 4.1 XGrammar 的核心创新

**上下文无关预填充（Context-Independent Preprocessing）**：

观察：在 FSM 中，许多状态的合法 token 集合不依赖于动态上下文（如当前生成的具体字符串），只依赖于 FSM 状态本身。

XGrammar 在编译阶段提前计算所有"上下文无关"状态的 token mask，大幅减少推理时的实时计算量。

**Adaptive Token Mask**：
- 对于简单状态（如"必须生成 0-9 之间的数字"）：使用预计算的 mask
- 对于复杂状态（依赖动态生成内容，如字符串字面量的内容）：实时计算

### 4.2 性能优势

| 方法 | 简单 Schema（<10 字段） | 复杂 Schema（>50 字段） |
|------|---------------------|---------------------|
| Outlines | ~1ms | ~50ms |
| XGrammar | <0.1ms | ~2ms |

对于批量推理，XGrammar 可以在 GPU 上并行化 token mask 的应用，进一步提升性能。

---

## 五、常见约束类型

### 5.1 JSON Schema 约束

```python
# vLLM 中使用 JSON Schema 约束
from vllm import LLM, SamplingParams

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"]
}

sampling_params = SamplingParams(
    temperature=0.8,
    guided_decoding={"json": schema}
)

outputs = llm.generate("介绍一下李明", sampling_params)
# 输出保证是合法的 JSON，且符合 schema
```

### 5.2 正则表达式约束

```python
# 约束输出为手机号格式
sampling_params = SamplingParams(
    guided_decoding={"regex": r"1[3-9]\d{9}"}
)

# 约束输出为日期格式
sampling_params = SamplingParams(
    guided_decoding={"regex": r"\d{4}-\d{2}-\d{2}"}
)
```

### 5.3 枚举约束（选择题）

```python
# 约束输出为 A/B/C/D 之一
sampling_params = SamplingParams(
    guided_decoding={"choice": ["A", "B", "C", "D"]}
)
```

### 5.4 Grammar（CFG）约束

CFG（上下文无关文法）约束比正则表达式更强大，可以描述递归结构（如嵌套括号、代码语法）：

```python
# SQL 语法约束（EBNF 格式）
sql_grammar = r"""
?start: select_stmt
select_stmt: "SELECT" column_list "FROM" table_name ("WHERE" condition)?
column_list: "*" | column_name ("," column_name)*
...
"""

sampling_params = SamplingParams(
    guided_decoding={"grammar": sql_grammar}
)
```

---

## 六、各框架实现对比

### 6.1 vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...", guided_decoding_backend="outlines")  # 或 "xgrammar"

# JSON Schema
SamplingParams(guided_decoding={"json": schema})

# 正则表达式
SamplingParams(guided_decoding={"regex": pattern})

# 选择
SamplingParams(guided_decoding={"choice": ["yes", "no"]})

# Grammar
SamplingParams(guided_decoding={"grammar": grammar_str})
```

### 6.2 SGLang

```python
import sglang as sgl

# JSON Schema（使用 XGrammar）
@sgl.function
def extract_info(s, text):
    s += sgl.user(f"从以下文本提取信息：{text}")
    s += sgl.assistant(
        sgl.gen("info",
                json_schema=schema,  # JSON Schema 约束
                max_tokens=200)
    )

# 正则表达式
sgl.gen("output", regex=r"\d{4}")

# 选择题
sgl.gen("answer", choices=["A", "B", "C", "D"])
```

### 6.3 OpenAI API

```python
from openai import OpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "描述张三，30岁"}],
    response_format=Person,
)
person = response.choices[0].message.parsed
```

---

## 七、约束解码的性能开销

### 7.1 主要开销来源

1. **Token Mask 计算**：每个 decode step 确定合法 token 集合
2. **Mask 应用**：将 mask 应用到 logits（向量操作，GPU 高效）
3. **FSM 状态更新**：维护每个请求的 FSM 状态

### 7.2 批量推理中的挑战

- 不同请求有不同约束，难以批量化 mask 计算
- 需要为每个请求独立维护 FSM 状态
- 解决：XGrammar 的预计算 mask 可以高效处理异构约束批次

### 7.3 实测开销

以 Llama-3-8B，batch=32 为例：
- 无约束：基准
- JSON Schema 约束（Outlines）：+15-30% 延迟
- JSON Schema 约束（XGrammar）：+2-5% 延迟

---

## 八、约束解码 vs 后处理

很多工程实践中，开发者会让 LLM 生成自由文本，然后通过后处理（正则提取、JSON 解析、重试等）来获得结构化结果。与之相比：

| | 约束解码 | 后处理 |
|--|--------|--------|
| 成功率 | 100%（数学保证） | 90-99%（可能失败） |
| 性能开销 | 轻微增加（mask 计算） | 可能需要多次重试 |
| 实现复杂度 | 需要框架支持 | 简单 |
| 灵活性 | 需要预先定义约束 | 后处理逻辑灵活 |

对于生产系统，约束解码是更可靠的选择，特别是当 schema 复杂、不容许任何解析错误时。

---

## 九、总结

约束解码是 LLM 工程应用的关键技术：

1. **核心机制**：Logit Masking，通过设置非法 token 的 logit 为 $-\infty$ 实现
2. **约束表达**：FSM（正则/JSON Schema）或 CFG（复杂语法）
3. **高效实现**：XGrammar 通过预计算大幅降低推理开销
4. **支持约束类型**：JSON Schema、正则表达式、枚举、CFG（SQL、代码等）
5. **框架支持**：vLLM（Outlines/XGrammar）、SGLang（XGrammar，原生集成）
6. **应用价值**：100% 保证输出格式合法，消除解析失败和重试

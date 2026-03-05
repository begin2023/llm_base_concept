# 6. Function Call 与 Agent 详解

---

## 一、Function Call（函数调用）

### 1.1 核心概念

Function Call（函数调用）是大语言模型（LLM）与外部世界交互的关键机制。其核心思想是：**模型本身不执行函数，而是通过生成结构化的输出（通常是 JSON），来"告诉"调用方应该调用哪个函数、传入什么参数**。调用方（通常是应用层代码）解析模型的输出，执行对应的函数，再将结果返回给模型进行下一步推理。

传统的 LLM 只能生成自然语言文本，面对"今天北京天气如何"这类问题，只能根据训练数据给出可能过时或编造的答案。有了 Function Call，模型可以：
- 识别出用户意图需要调用外部 API（比如天气查询接口）
- 生成结构化的函数调用请求（函数名 + 参数）
- 等待调用方执行函数后返回结果
- 基于真实的函数返回值生成最终回答

**本质上，Function Call 将 LLM 从一个"只会说话的模型"升级为一个"能操作工具的模型"。**

### 1.2 实现原理

#### 1.2.1 训练阶段

Function Call 能力不是凭空出现的，需要在模型训练（通常是 SFT/RLHF 阶段）中专门注入：

**（1）函数描述的注入方式**

在训练数据中，将函数的 schema 描述作为 system prompt 或特殊 token 的一部分注入。例如：

```json
{
  "name": "get_weather",
  "description": "获取指定城市的当前天气信息",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称，例如'北京'"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "温度单位"
      }
    },
    "required": ["city"]
  }
}
```

**（2）训练数据格式**

训练数据包含三种类型的样本：
- **直接回答**：用户问题不需要调用函数，模型直接回答
- **函数调用**：用户问题需要调用函数，模型输出函数调用 JSON
- **函数结果处理**：模型接收函数返回值，生成最终回答

训练数据示例：
```
User: 北京今天天气怎么样？
Assistant: {"function_call": {"name": "get_weather", "arguments": {"city": "北京"}}}
Function Result: {"temperature": 25, "condition": "晴", "humidity": 40}
Assistant: 北京今天天气晴朗，气温25°C，湿度40%，非常适合户外活动。
```

**（3）特殊 Token**

许多模型使用特殊 token 来标记函数调用的边界：
- `<|function_call|>` / `<|end_function_call|>`
- `<tool_call>` / `</tool_call>`（如 Qwen 系列）
- `<|plugin|>` / `<|/plugin|>`

这些特殊 token 在 tokenizer 中有对应的 token id，模型通过学习这些 token 的使用模式来掌握何时以及如何调用函数。

#### 1.2.2 推理阶段

推理时的 Function Call 流程：

```
用户输入 + 函数描述（tools）
        ↓
    模型推理（LLM）
        ↓
   ┌─────────────────────┐
   │ 判断是否需要调用函数  │
   └─────────────────────┘
        ↓              ↓
   不需要调用         需要调用
        ↓              ↓
   直接生成回答    生成函数调用JSON
                       ↓
                  应用层解析JSON
                       ↓
                  执行对应函数
                       ↓
                  获取函数返回值
                       ↓
              将返回值拼接回上下文
                       ↓
                  模型继续推理
                       ↓
                  生成最终回答
```

**关键点**：模型的输出本质上还是 token 序列，只是这些 token 恰好组成了合法的 JSON 格式。推理引擎（如 vLLM、SGLang）需要能够：
1. 识别输出中的函数调用标记
2. 在函数调用完成时停止生成（stop token）
3. 支持流式输出中的函数调用检测

#### 1.2.3 并行函数调用（Parallel Function Calling）

现代模型支持在一次推理中同时调用多个函数：

```json
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\": \"上海\"}"}}
  ]
}
```

这要求模型在训练时见过并行调用的样本，推理引擎需要能够解析多个函数调用。

### 1.3 OpenAI Function Calling 工作流程

OpenAI 的 Function Calling 是业界最具影响力的实现，其完整工作流程如下：

#### 1.3.1 API 请求结构

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "北京和上海今天天气怎么样？"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    tool_choice="auto"  # auto / none / required / 指定函数
)
```

#### 1.3.2 完整交互流程（多轮）

```
第一轮请求：
  messages: [user: "北京和上海天气如何？"]
  tools: [get_weather 的定义]

第一轮响应：
  message: {
    role: "assistant",
    tool_calls: [
      {id: "call_abc", function: {name: "get_weather", arguments: '{"city":"北京"}'}},
      {id: "call_def", function: {name: "get_weather", arguments: '{"city":"上海"}'}}
    ]
  }
  finish_reason: "tool_calls"

客户端执行函数，获取结果。

第二轮请求：
  messages: [
    user: "北京和上海天气如何？",
    assistant: {tool_calls: [...]},     # 将第一轮的 assistant 消息原样放回
    tool: {tool_call_id: "call_abc", content: '{"temp":25,"condition":"晴"}'},
    tool: {tool_call_id: "call_def", content: '{"temp":28,"condition":"多云"}'}
  ]

第二轮响应：
  message: {
    role: "assistant",
    content: "北京今天25°C，晴天；上海28°C，多云。"
  }
  finish_reason: "stop"
```

#### 1.3.3 tool_choice 参数详解

| 值 | 含义 |
|---|---|
| `"auto"` | 模型自行决定是否调用函数（默认值） |
| `"none"` | 禁止模型调用任何函数 |
| `"required"` | 强制模型必须调用至少一个函数 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定函数 |

#### 1.3.4 Structured Outputs 模式

OpenAI 后来引入了 `strict: true` 模式，确保函数调用的参数严格符合 JSON Schema：

```python
tools=[{
    "type": "function",
    "function": {
        "name": "get_weather",
        "strict": True,  # 启用严格模式
        "parameters": { ... }
    }
}]
```

严格模式下，推理引擎使用 **Constrained Decoding**（约束解码）技术，在每一步 token 生成时限制候选 token 集合，确保输出一定是合法 JSON。这对推理引擎有额外的计算开销，但保证了输出格式的正确性。

### 1.4 Function Call 在推理引擎中的实现

#### 1.4.1 vLLM 的实现

vLLM 通过 `--tool-call-parser` 参数指定解析器：

```bash
vllm serve model_name \
  --tool-call-parser hermes \  # 或 mistral、llama3、internlm 等
  --enable-auto-tool-choice
```

不同模型使用不同的函数调用格式，vLLM 内置了多种解析器来处理：
- **Hermes 格式**：`<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
- **Mistral 格式**：`[TOOL_CALLS] [{"name":"...", "arguments":{...}}]`
- **Llama3 格式**：使用 `<|python_tag|>` 等特殊 token

#### 1.4.2 SGLang 的实现

SGLang 同样支持工具调用，并且在结构化输出方面有更好的性能（使用 xgrammar 引擎）：

```python
# SGLang 支持通过 JSON schema 约束输出
@sgl.function
def tool_use(s, question, tools):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("response", tools=tools))
```

#### 1.4.3 流式输出中的 Function Call

流式场景下，Function Call 的处理更复杂：

```
chunk 1: {"tool_calls": [{"index": 0, "function": {"name": "get_"}}]}
chunk 2: {"tool_calls": [{"index": 0, "function": {"name": "weather"}}]}
chunk 3: {"tool_calls": [{"index": 0, "function": {"arguments": "{\"ci"}}]}
chunk 4: {"tool_calls": [{"index": 0, "function": {"arguments": "ty\": \"北京\"}"}}]}
```

推理引擎需要：
1. 实时检测输出是否进入了函数调用模式
2. 增量式地输出函数名和参数的 delta
3. 在函数调用结束时正确设置 `finish_reason: "tool_calls"`

---

## 二、Agent（智能体）

### 2.1 核心概念

Agent（智能体）是基于大语言模型构建的**自主决策和执行系统**。与简单的 LLM 调用不同，Agent 具有以下核心特征：

1. **自主性（Autonomy）**：能够自主决定下一步行动，而非被动等待指令
2. **工具使用（Tool Use）**：能够调用外部工具和 API 来完成任务
3. **记忆（Memory）**：具有短期记忆（上下文窗口）和长期记忆（外部存储）
4. **规划（Planning）**：能够将复杂任务分解为子任务并制定执行计划
5. **反思（Reflection）**：能够评估自身行动的结果并进行调整

**一个简洁的定义：Agent = LLM + Memory + Planning + Tool Use**

### 2.2 Agent 核心架构

#### 2.2.1 ReAct（Reasoning + Acting）

ReAct 是最经典的 Agent 架构之一，由 Yao et al. (2022) 提出。核心思想是让模型交替进行**推理（Reasoning）**和**行动（Acting）**。

**工作流程**：

```
循环开始：
  ┌──────────────────────────┐
  │ Thought（思考）           │ → 模型分析当前状态，思考下一步该做什么
  │ Action（行动）            │ → 模型决定调用哪个工具，传入什么参数
  │ Observation（观察）       │ → 工具返回结果，模型观察结果
  └──────────────────────────┘
  是否得到最终答案？
    → 否：继续循环
    → 是：输出 Final Answer
```

**具体示例**：

```
User: "特斯拉2024年Q4的营收是多少？与Q3相比增长了多少？"

Thought 1: 我需要先查询特斯拉2024年Q4的营收数据。
Action 1: search_financial_data(company="Tesla", quarter="2024Q4", metric="revenue")
Observation 1: Tesla 2024 Q4 revenue: $25.7 billion

Thought 2: 现在我有了Q4数据，还需要Q3数据来计算增长率。
Action 2: search_financial_data(company="Tesla", quarter="2024Q3", metric="revenue")
Observation 2: Tesla 2024 Q3 revenue: $25.2 billion

Thought 3: 我现在有了两个季度的数据，可以计算增长率了。
增长率 = (25.7 - 25.2) / 25.2 × 100% = 1.98%

Final Answer: 特斯拉2024年Q4营收为257亿美元，与Q3的252亿美元相比，环比增长约2.0%。
```

**ReAct 的优势**：
- 推理过程可解释、可追踪
- 每一步都有明确的思考链，便于调试
- 适合需要多步推理和工具调用的场景

**ReAct 的局限**：
- 串行执行，效率较低
- 容易陷入循环（反复做同样的事情）
- 对模型的推理能力要求较高

#### 2.2.2 Plan-and-Execute（计划与执行）

Plan-and-Execute 架构将**规划**和**执行**分离为两个阶段：

```
┌─────────────────────────────────┐
│          Planner（规划器）        │
│  输入：用户任务                   │
│  输出：步骤列表 [Step1, Step2, ...]│
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│         Executor（执行器）        │
│  逐步执行 Planner 生成的步骤      │
│  每步可以调用工具或子Agent         │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│        Re-Planner（重规划器）     │
│  根据执行结果决定是否需要调整计划   │
└─────────────────────────────────┘
```

**示例**：

```
User: "帮我写一篇关于量子计算最新进展的技术博客"

Planner 输出:
  Step 1: 搜索2024年量子计算领域的重大突破
  Step 2: 搜索主要科技公司（Google、IBM、Microsoft）的量子计算最新进展
  Step 3: 搜索量子纠错领域的最新论文
  Step 4: 整理搜索结果，提取关键信息
  Step 5: 撰写博客大纲
  Step 6: 撰写博客正文
  Step 7: 校对和润色

Executor:
  执行 Step 1 → 调用 web_search 工具 → 获取结果
  执行 Step 2 → 调用 web_search 工具 → 获取结果
  ...

Re-Planner:
  检查 Step 1-3 的结果，发现量子纠错方面信息不足
  → 追加 Step 3.5: 搜索 arXiv 上的量子纠错论文
```

**优势**：
- 适合复杂、多步骤任务
- 规划和执行分离，更容易管理
- 支持动态重规划

**局限**：
- 初始规划可能不准确
- 规划本身消耗 LLM 推理资源
- 实现复杂度较高

#### 2.2.3 Tool Use Loop（工具使用循环）

这是最简单也是最常用的 Agent 架构，本质上就是 Function Call 的循环调用：

```python
def agent_loop(user_message, tools, max_iterations=10):
    messages = [{"role": "user", "content": user_message}]

    for i in range(max_iterations):
        # 调用 LLM
        response = llm.chat(messages=messages, tools=tools)

        # 如果模型没有调用工具，说明任务完成
        if response.finish_reason == "stop":
            return response.content

        # 如果模型调用了工具
        if response.finish_reason == "tool_calls":
            messages.append(response.message)  # 添加 assistant 消息

            for tool_call in response.tool_calls:
                # 执行工具
                result = execute_tool(tool_call.function.name,
                                     tool_call.function.arguments)
                # 将结果添加到消息列表
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

    return "达到最大迭代次数"
```

**这种架构的关键特点**：
- 简单直接，易于实现
- 模型自主决定何时调用工具、何时停止
- 每一轮迭代都将历史消息（包括工具结果）传给模型
- 通过 `max_iterations` 防止无限循环

### 2.3 Agent 记忆系统

#### 2.3.1 短期记忆（Short-term Memory）

就是 LLM 的上下文窗口。所有的对话历史、工具调用结果都在这里。

**问题**：上下文窗口有限（即使 128K token），长时间运行的 Agent 会超出窗口限制。

**解决方案**：
- 滑动窗口：丢弃最早的消息
- 摘要压缩：定期对历史消息做摘要
- 重要性筛选：只保留关键信息

#### 2.3.2 长期记忆（Long-term Memory）

使用外部存储持久化重要信息：

```
┌─────────────┐     ┌──────────────────┐
│   Agent      │────→│  向量数据库       │  语义检索历史经验
│   (LLM)      │←────│  (Milvus/Chroma)  │
│              │────→│  关系型数据库      │  结构化知识存储
│              │←────│  (PostgreSQL)     │
│              │────→│  文件系统         │  长文本/文档存储
│              │←────│  (Local/S3)      │
└─────────────┘     └──────────────────┘
```

### 2.4 多 Agent 协作框架

#### 2.4.1 AutoGen（微软）

AutoGen 是微软推出的多 Agent 对话框架，核心理念是**多个 Agent 之间通过对话协作**完成任务。

**架构特点**：

```
┌──────────────┐    消息     ┌──────────────┐
│  UserProxy    │ ←────────→ │  Assistant    │
│  Agent        │            │  Agent        │
│ (执行代码)    │            │ (生成代码)    │
└──────────────┘            └──────────────┘
                                    ↑
                                    │ 消息
                                    ↓
                            ┌──────────────┐
                            │   Critic     │
                            │   Agent      │
                            │  (审查代码)   │
                            └──────────────┘
```

**核心概念**：
- **ConversableAgent**：所有 Agent 的基类，支持发送和接收消息
- **AssistantAgent**：基于 LLM 的助手，负责推理和生成
- **UserProxyAgent**：代理用户行为，可以执行代码、提供人类反馈
- **GroupChat**：多个 Agent 的群聊，支持自动选择下一个发言者

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 创建 Agent
coder = AssistantAgent("coder", llm_config=llm_config,
    system_message="你是一个Python专家，负责编写代码。")
reviewer = AssistantAgent("reviewer", llm_config=llm_config,
    system_message="你是一个代码审查专家，负责审查代码质量。")
executor = UserProxyAgent("executor",
    code_execution_config={"work_dir": "coding"})

# 创建群聊
group_chat = GroupChat(agents=[coder, reviewer, executor], messages=[])
manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# 启动对话
executor.initiate_chat(manager, message="写一个快速排序算法并测试")
```

#### 2.4.2 CrewAI

CrewAI 采用**角色扮演**的方式组织多 Agent 协作，更接近现实中的团队协作模式。

**核心概念**：
- **Agent（代理人）**：有角色（role）、目标（goal）、背景故事（backstory）
- **Task（任务）**：具体的工作单元，分配给特定的 Agent
- **Crew（团队）**：一组 Agent 和 Task 的组合
- **Process（流程）**：顺序执行（sequential）或层级执行（hierarchical）

```python
from crewai import Agent, Task, Crew, Process

# 定义 Agent
researcher = Agent(
    role="高级研究分析师",
    goal="发现AI领域的最新技术趋势",
    backstory="你在顶级科技研究机构工作了20年...",
    tools=[search_tool, arxiv_tool],
    llm=llm
)
writer = Agent(
    role="技术作家",
    goal="撰写引人入胜的技术文章",
    backstory="你是一位获奖的技术博主...",
    llm=llm
)

# 定义任务
research_task = Task(
    description="研究2024年AI最重要的5个突破",
    agent=researcher,
    expected_output="一份详细的研究报告"
)
writing_task = Task(
    description="基于研究报告撰写一篇2000字的博客文章",
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出
    expected_output="一篇完整的博客文章"
)

# 创建团队并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # 顺序执行
)
result = crew.kickoff()
```

#### 2.4.3 LangGraph（LangChain）

LangGraph 将 Agent 的工作流建模为**有向图（Graph）**，每个节点是一个处理步骤，边是条件跳转。

```python
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: list
    next_action: str

# 定义节点（处理函数）
def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state):
    tool_result = execute_tool(state["messages"][-1])
    return {"messages": [tool_result]}

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return END

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tool", call_tool)
graph.add_edge("tool", "agent")  # 工具执行后回到 agent
graph.add_conditional_edges("agent", should_continue)  # 条件分支
graph.set_entry_point("agent")

app = graph.compile()
```

#### 2.4.4 框架对比

| 特性 | AutoGen | CrewAI | LangGraph |
|------|---------|--------|-----------|
| 协作模式 | 对话驱动 | 角色扮演 | 图工作流 |
| 灵活性 | 高 | 中 | 非常高 |
| 易用性 | 中 | 高 | 低 |
| 适用场景 | 代码生成、讨论 | 内容创作、研究 | 复杂工作流 |
| 状态管理 | 消息历史 | 任务上下文 | 显式状态图 |
| 人在回路 | 原生支持 | 支持 | 支持 |

### 2.5 与推理引擎的关系

Agent 和 Function Call 对推理引擎（如 vLLM、SGLang、TensorRT-LLM）提出了特殊的需求：

#### 2.5.1 流式输出支持

Agent 场景下，流式输出（Streaming）至关重要：
- 用户需要实时看到 Agent 的思考过程（Thought）
- 函数调用需要在模型输出特定 token 时立即触发
- 推理引擎需要支持 **SSE（Server-Sent Events）** 协议

```
# 流式输出中的 Function Call 检测
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"北京\"}"}}]}}]}
data: {"choices":[{"finish_reason":"tool_calls"}]}
```

#### 2.5.2 结构化输出保证

Function Call 本质上要求模型输出合法的 JSON。推理引擎可以通过以下方式保证：

- **Guided Decoding / Constrained Decoding**：在每步 token 采样时，根据 JSON Schema 限制候选 token
- **JSON Mode**：强制输出合法 JSON（但不保证符合特定 schema）
- **Grammar-based Sampling**：基于形式化语法（如 EBNF）约束输出

vLLM 和 SGLang 都支持这些功能：

```python
# vLLM 的 guided decoding
response = client.chat.completions.create(
    model="model_name",
    messages=[...],
    extra_body={
        "guided_json": json_schema,  # JSON Schema 约束
        # 或
        "guided_grammar": ebnf_grammar  # EBNF 语法约束
    }
)
```

#### 2.5.3 长上下文支持

Agent 的多轮交互会导致上下文不断增长（每轮的 Thought、Action、Observation 都会被追加），推理引擎需要：

- **高效的 KV Cache 管理**：重用之前轮次的 KV Cache，避免重复计算
- **Prefix Caching**：多轮对话中，前面的消息是固定前缀，可以缓存
- **长上下文推理优化**：如 Ring Attention、Flash Attention 等

#### 2.5.4 低延迟要求

Agent 的每一步都需要一次 LLM 推理，多步骤任务的总延迟 = 单步延迟 × 步骤数。因此：

- **TTFT（Time to First Token）**：Agent 场景更关注首 token 延迟
- **投机解码**：可以显著降低单步推理延迟
- **批处理**：当多个 Agent 并行运行时，推理引擎的批处理能力很重要

#### 2.5.5 Stop Token 和特殊 Token 处理

推理引擎需要正确处理各种 stop 条件：
- 模型输出 `</tool_call>` 时停止生成
- 模型输出 `\nObservation:` 时停止（ReAct 格式）
- 支持自定义 stop sequences

```python
response = client.chat.completions.create(
    model="model_name",
    messages=[...],
    stop=["Observation:", "</tool_call>", "\nHuman:"]
)
```

### 2.6 前沿趋势

1. **Code Agent**：Agent 通过编写和执行代码来解决问题（如 OpenAI Code Interpreter、Claude Code）
2. **Computer Use Agent**：Agent 直接操作计算机界面（如 Claude Computer Use、Open Interpreter）
3. **Multi-modal Agent**：结合视觉、语音等多模态能力的 Agent
4. **Agent as OS**：将 Agent 构建为操作系统级别的应用，管理文件、进程、网络等
5. **Self-evolving Agent**：能够自我改进 prompt 和工具使用策略的 Agent

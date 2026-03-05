# 7. MCP（Model Context Protocol）详解

---

## 一、概念与背景

### 1.1 什么是 MCP

MCP（Model Context Protocol，模型上下文协议）是 **Anthropic 于 2024 年 11 月提出并开源的一项标准化协议**，旨在为大语言模型（LLM）与外部数据源、工具之间的交互提供**统一的、开放的接口标准**。

用一个类比来理解：**MCP 之于 LLM，就像 USB 之于计算机外设。** 在 USB 出现之前，每种外设（键盘、鼠标、打印机）都有自己的接口标准，连接非常混乱。USB 统一了接口标准后，任何设备都可以即插即用。MCP 的目标也是如此——让任何 LLM 应用都能通过统一的协议接入任何数据源和工具。

### 1.2 为什么需要 MCP

在 MCP 出现之前，LLM 与外部工具的集成面临严重的碎片化问题：

**问题一：N×M 集成问题**

假设有 N 个 LLM 应用（ChatGPT、Claude、自建应用）和 M 个工具/数据源（GitHub、Slack、数据库、搜索引擎），每个应用要接入每个工具都需要单独开发集成代码，总共需要 N×M 个集成。

```
传统方式（N×M 集成）：

  应用A ──→ GitHub 专用集成
  应用A ──→ Slack 专用集成
  应用A ──→ 数据库专用集成
  应用B ──→ GitHub 专用集成（重新开发）
  应用B ──→ Slack 专用集成（重新开发）
  应用B ──→ 数据库专用集成（重新开发）
  ...

MCP 方式（N+M 集成）：

  应用A ──→ MCP Client ──→ MCP Server (GitHub)
  应用B ──→ MCP Client ──→ MCP Server (Slack)
                           MCP Server (数据库)
                           MCP Server (搜索引擎)
```

通过 MCP，应用只需要实现一次 MCP Client，工具只需要实现一次 MCP Server，集成数量从 N×M 降到 N+M。

**问题二：Function Call 的局限性**

虽然 OpenAI 的 Function Calling 解决了 LLM 调用工具的基本问题，但它有明显局限：
- 工具定义与调用耦合在 API 请求中，没有独立的发现机制
- 每个应用需要自行管理工具的生命周期
- 缺乏标准化的资源访问方式
- 没有统一的权限和安全模型
- 不支持动态发现和注册工具

**问题三：上下文孤岛**

LLM 应用的数据散落在各处（本地文件、云服务、数据库），缺乏统一的方式让模型访问这些分散的上下文信息。

### 1.3 设计目标

MCP 的核心设计目标：
1. **标准化**：提供统一的协议规范，任何人都可以基于此开发 Server 和 Client
2. **解耦**：工具/数据源的实现与 LLM 应用完全解耦
3. **安全**：内置权限控制和安全机制
4. **可发现**：支持工具和资源的动态发现
5. **可扩展**：易于添加新的工具和数据源
6. **本地优先**：支持本地运行，数据不必经过第三方服务器

---

## 二、核心架构

### 2.1 整体架构

MCP 采用经典的 **Client-Server 架构**，分为三个核心层：

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Application                        │
│                    (Claude Desktop, IDE, etc.)                │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │         │
│  │     #1      │  │     #2      │  │     #3      │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │
     Transport        Transport       Transport
     (stdio)         (HTTP/SSE)      (stdio)
          │               │               │
  ┌───────┴──────┐ ┌──────┴───────┐ ┌────┴────────┐
  │  MCP Server  │ │  MCP Server  │ │  MCP Server  │
  │  (GitHub)    │ │  (Database)  │ │  (FileSystem)│
  │              │ │              │ │              │
  │  Tools       │ │  Tools       │ │  Resources   │
  │  Resources   │ │  Resources   │ │  Prompts     │
  │  Prompts     │ │  Prompts     │ │              │
  └──────────────┘ └──────────────┘ └──────────────┘
```

### 2.2 MCP Server（服务端）

MCP Server 是**工具和数据的提供者**，负责暴露三类核心能力：

#### 2.2.1 Tools（工具）

工具是 MCP 最核心的能力，允许 LLM 通过 Server 执行操作。

```json
{
  "name": "create_github_issue",
  "description": "在 GitHub 仓库中创建一个新的 Issue",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": {
        "type": "string",
        "description": "仓库名称，格式为 owner/repo"
      },
      "title": {
        "type": "string",
        "description": "Issue 标题"
      },
      "body": {
        "type": "string",
        "description": "Issue 内容（Markdown 格式）"
      },
      "labels": {
        "type": "array",
        "items": {"type": "string"},
        "description": "标签列表"
      }
    },
    "required": ["repo", "title"]
  }
}
```

**Tool 与 Function Call 的对应关系**：MCP Tool 的 schema 会被转换为 LLM 的 Function Call 定义，LLM 生成的函数调用会被 MCP Client 路由到对应的 MCP Server 执行。

#### 2.2.2 Resources（资源）

资源是 MCP Server 向 LLM 暴露的**数据**，类似于 REST API 中的 GET 端点。资源是只读的，LLM 可以读取但不能修改。

```json
{
  "uri": "github://owner/repo/issues/42",
  "name": "Issue #42: Fix memory leak",
  "mimeType": "text/markdown",
  "description": "GitHub Issue 的详细内容"
}
```

资源使用 URI 标识，支持：
- **静态资源**：固定 URI，如 `config://app/settings`
- **动态资源**：模板化 URI，如 `github://{owner}/{repo}/issues/{id}`
- **资源列表**：Server 可以列出所有可用资源
- **订阅变更**：Client 可以订阅资源变更通知

#### 2.2.3 Prompts（提示模板）

Server 可以暴露预定义的 prompt 模板，帮助 LLM 更好地使用该 Server 的功能。

```json
{
  "name": "review_code",
  "description": "代码审查提示模板",
  "arguments": [
    {
      "name": "language",
      "description": "编程语言",
      "required": true
    },
    {
      "name": "code",
      "description": "要审查的代码",
      "required": true
    }
  ]
}
```

当用户选择此 prompt 模板并填入参数后，Server 返回完整的 prompt：

```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "请审查以下 Python 代码，重点关注：\n1. 代码风格\n2. 潜在 bug\n3. 性能问题\n\n```python\ndef foo(): ...\n```"
      }
    }
  ]
}
```

### 2.3 MCP Client（客户端）

MCP Client 运行在**宿主应用（Host Application）**中，负责：

1. **连接管理**：与 MCP Server 建立和维护连接
2. **能力发现**：发现 Server 提供的 Tools、Resources、Prompts
3. **请求路由**：将 LLM 的工具调用请求路由到正确的 Server
4. **协议转换**：将 MCP 协议的结果转换为 LLM 可以理解的格式

**一个 Host Application 可以同时连接多个 MCP Server**，每个 Server 通过独立的 MCP Client 实例管理。

### 2.4 Transport 层（传输层）

MCP 支持多种传输方式：

#### 2.4.1 Stdio（标准输入输出）

```
Host Application ←→ stdin/stdout ←→ MCP Server（子进程）
```

- Server 作为 Host 的子进程运行
- 通过标准输入/输出管道通信
- **最常用于本地 Server**
- 延迟低，无需网络

配置示例（Claude Desktop 的 `claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxx"
      }
    }
  }
}
```

#### 2.4.2 HTTP + SSE（Server-Sent Events）

```
Host Application ──HTTP POST──→ MCP Server（远程）
Host Application ←──SSE────── MCP Server（远程）
```

- Client 通过 HTTP POST 发送请求
- Server 通过 SSE 推送响应和通知
- **适用于远程 Server**
- 支持跨网络访问

#### 2.4.3 Streamable HTTP（2025年新增）

这是 MCP 规范后来引入的改进传输方式，结合了 HTTP 和 SSE 的优势：
- 单个 HTTP 端点同时支持请求-响应和流式推送
- 更简单的部署和防火墙友好
- 支持无状态和有状态两种模式

### 2.5 协议消息格式

MCP 基于 **JSON-RPC 2.0** 协议进行消息通信：

**请求消息**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "create_github_issue",
    "arguments": {
      "repo": "owner/repo",
      "title": "Bug: Memory leak in cache module"
    }
  }
}
```

**响应消息**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Successfully created issue #43"
      }
    ]
  }
}
```

**通知消息**（无需响应）：
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "github://owner/repo/issues/43"
  }
}
```

### 2.6 协议生命周期

```
1. 初始化阶段（Initialize）
   Client ──→ initialize 请求 ──→ Server
   Client ←── 支持的能力列表 ←── Server
   Client ──→ initialized 通知 ──→ Server

2. 能力发现阶段
   Client ──→ tools/list ──→ Server
   Client ←── 工具列表 ←── Server
   Client ──→ resources/list ──→ Server
   Client ←── 资源列表 ←── Server
   Client ──→ prompts/list ──→ Server
   Client ←── 提示模板列表 ←── Server

3. 正常操作阶段
   Client ──→ tools/call ──→ Server
   Client ←── 调用结果 ←── Server
   Client ──→ resources/read ──→ Server
   Client ←── 资源内容 ←── Server

4. 关闭阶段
   Client ──→ 关闭连接 ──→ Server
```

---

## 三、与 Function Call 的区别和联系

### 3.1 联系

MCP 和 Function Call **不是竞争关系，而是互补和层次关系**：

```
┌──────────────────────────────────────────────┐
│              应用层（Agent / App）              │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │       MCP Client（协议层）           │     │
│  │  - 发现和管理工具                     │     │
│  │  - 路由工具调用请求                   │     │
│  │  - 管理资源和提示模板                  │     │
│  └──────────────┬──────────────────────┘     │
│                 │                             │
│  ┌──────────────┴──────────────────────┐     │
│  │    Function Call（模型层）            │     │
│  │  - LLM 生成工具调用的 JSON            │     │
│  │  - 模型侧的工具选择和参数生成          │     │
│  └─────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

**Function Call 是模型层面的能力**：模型能够理解工具描述并生成结构化的调用请求。
**MCP 是协议层面的标准**：定义了工具如何被发现、注册、调用和管理。

实际工作流：
1. MCP Client 从 MCP Server 获取可用工具列表
2. 将工具列表转换为 LLM 的 Function Call 格式传给模型
3. 模型通过 Function Call 机制生成工具调用
4. MCP Client 将调用请求路由到对应的 MCP Server 执行
5. MCP Server 返回结果，传回给模型继续推理

### 3.2 核心区别

| 维度 | Function Call | MCP |
|------|--------------|-----|
| **层次** | 模型能力层 | 协议标准层 |
| **定义方** | 各 LLM 提供商各自定义 | Anthropic 提出的开放标准 |
| **工具发现** | 需要应用手动管理 | 协议内置动态发现机制 |
| **工具定义** | API 请求中传入 | Server 注册时声明 |
| **传输方式** | 内嵌在 LLM API 请求中 | 独立的 JSON-RPC 通信 |
| **标准化** | 各家格式不同 | 统一的开放标准 |
| **资源管理** | 不涉及 | 内置资源抽象 |
| **安全模型** | 由应用自行管理 | 协议层面的权限控制 |
| **生态系统** | 工具与应用强耦合 | 工具与应用完全解耦 |

---

## 四、在推理部署中的应用场景

### 4.1 本地工具集成

最典型的场景是在 IDE 中集成本地开发工具：

```
┌─────────────────────────────────────┐
│          IDE（如 VS Code + Cursor）   │
│                                     │
│  ┌───────┐   ┌───────────────────┐  │
│  │ LLM   │←→│ MCP Client        │  │
│  │ Agent │   │                   │  │
│  └───────┘   └───┬───┬───┬──────┘  │
│                  │   │   │          │
└──────────────────┼───┼───┼──────────┘
                   │   │   │
          ┌────────┘   │   └────────┐
          ↓            ↓            ↓
   MCP Server     MCP Server    MCP Server
   (文件系统)     (Git)         (数据库)
   - 读写文件     - 提交代码     - 查询数据
   - 搜索文件     - 查看diff     - 执行SQL
```

### 4.2 企业数据集成

企业场景中，MCP 可以安全地让 LLM 访问内部系统：

```
┌────────────────────────────────────────┐
│            企业 AI 平台                  │
│                                        │
│  ┌──────────────────┐                  │
│  │  MCP Client Hub  │                  │
│  └───┬───┬───┬──────┘                  │
│      │   │   │                         │
└──────┼───┼───┼─────────────────────────┘
       │   │   │
   ┌───┘   │   └───┐
   ↓       ↓       ↓
MCP Server  MCP Server  MCP Server
(CRM系统)  (ERP系统)   (知识库)
- 客户查询  - 订单管理  - 文档搜索
- 工单创建  - 库存查询  - FAQ检索
```

### 4.3 推理引擎集成

MCP 与推理引擎（vLLM、SGLang 等）的集成方式：

```
用户请求
    ↓
API Gateway / 应用服务器
    ↓
┌─────────────────────────────────────┐
│          Agent 框架层                 │
│  (LangChain / LlamaIndex / 自研)     │
│                                     │
│  ┌───────────────────────────┐      │
│  │      MCP Client           │      │
│  │  - 工具发现和管理           │      │
│  │  - 调用路由和结果处理       │      │
│  └──────────┬────────────────┘      │
│             │                       │
│  ┌──────────┴────────────────┐      │
│  │      推理引擎调用           │      │
│  │  - Function Call 格式构建   │      │
│  │  - 结构化输出解析           │      │
│  │  - 流式响应处理             │      │
│  └──────────┬────────────────┘      │
└─────────────┼───────────────────────┘
              ↓
┌─────────────────────────┐
│  推理引擎（vLLM/SGLang）  │
│  - 模型推理               │
│  - Function Call 生成     │
│  - 结构化输出约束          │
└─────────────────────────┘
```

**关键集成点**：

1. **Tool Schema 转换**：MCP 的 tool schema 需要转换为推理引擎支持的 Function Call 格式
2. **流式处理**：推理引擎的流式输出需要实时检测 Function Call，触发 MCP Server 调用
3. **Prefix Caching**：MCP 工具的返回结果加入上下文后，推理引擎可以利用 Prefix Caching 加速后续推理
4. **结构化输出**：推理引擎的 Guided Decoding 可以保证 Function Call 参数的格式正确性

### 4.4 多模型路由

MCP 可以作为中间层，将不同类型的工具调用路由到最适合的模型：

```
用户请求
    ↓
MCP Client Hub
    ├── 代码生成任务 → 路由到 Code LLM（DeepSeek Coder）
    ├── 数学推理任务 → 路由到 Math LLM（Qwen Math）
    └── 通用任务    → 路由到通用 LLM（GPT-4/Claude）
```

---

## 五、MCP Server 开发

### 5.1 Python SDK 示例

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建 Server 实例
server = Server("my-weather-server")

# 注册工具列表
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的天气信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        )
    ]

# 实现工具调用
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # 调用实际的天气 API
        weather_data = await fetch_weather(city)
        return [TextContent(
            type="text",
            text=f"{city}：{weather_data['condition']}，{weather_data['temperature']}°C"
        )]
    raise ValueError(f"Unknown tool: {name}")

# 启动 Server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 5.2 TypeScript SDK 示例

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "my-server", version: "1.0.0" });

// 注册工具
server.tool("get_weather",
  { city: z.string().describe("城市名称") },
  async ({ city }) => {
    const data = await fetchWeather(city);
    return {
      content: [{
        type: "text",
        text: `${city}: ${data.condition}, ${data.temperature}°C`
      }]
    };
  }
);

// 注册资源
server.resource("config://app/settings", async (uri) => {
  return {
    contents: [{
      uri: uri.href,
      mimeType: "application/json",
      text: JSON.stringify(appSettings)
    }]
  };
});

// 启动
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## 六、生态现状（截至 2025 年初）

### 6.1 官方 MCP Server

Anthropic 官方和社区已经开发了大量 MCP Server：

| Server | 功能 | 类型 |
|--------|------|------|
| `@modelcontextprotocol/server-filesystem` | 本地文件系统读写 | 官方 |
| `@modelcontextprotocol/server-github` | GitHub 操作 | 官方 |
| `@modelcontextprotocol/server-postgres` | PostgreSQL 查询 | 官方 |
| `@modelcontextprotocol/server-slack` | Slack 消息收发 | 官方 |
| `@modelcontextprotocol/server-brave-search` | Brave 搜索引擎 | 官方 |
| `@modelcontextprotocol/server-puppeteer` | 浏览器自动化 | 官方 |
| `@modelcontextprotocol/server-sqlite` | SQLite 数据库 | 官方 |
| `@modelcontextprotocol/server-google-maps` | Google 地图 | 官方 |
| `server-memory` | 知识图谱记忆 | 官方 |

### 6.2 支持 MCP 的客户端

| 客户端 | 支持程度 |
|--------|---------|
| Claude Desktop | 完整支持（原生） |
| Claude Code（CLI） | 完整支持 |
| Cursor | 支持（IDE 集成） |
| Windsurf | 支持 |
| Continue（VS Code） | 支持 |
| Zed | 支持 |
| Cline | 支持 |
| 各种自建 Agent | 通过 SDK 支持 |

### 6.3 生态发展趋势

1. **Server 数量快速增长**：GitHub 上已有数千个社区开发的 MCP Server
2. **企业采纳加速**：越来越多的企业开始构建内部 MCP Server
3. **框架集成**：LangChain、LlamaIndex 等主流框架已支持 MCP
4. **标准化推进**：更多 LLM 提供商开始支持或兼容 MCP
5. **安全增强**：OAuth 2.0 集成、细粒度权限控制等安全特性不断完善
6. **远程 Server 生态**：Cloudflare、Vercel 等平台开始提供 MCP Server 托管服务

### 6.4 局限性与挑战

1. **性能开销**：每次工具调用都需要经过 JSON-RPC 通信，相比直接函数调用有额外延迟
2. **安全审计**：第三方 MCP Server 的安全性难以保证，存在数据泄露风险
3. **版本兼容**：协议仍在快速迭代中，Server 和 Client 之间可能存在版本不兼容
4. **调试困难**：多层协议嵌套使得问题排查变得复杂
5. **生态碎片化**：同一功能可能有多个 MCP Server 实现，质量参差不齐
6. **与现有 Function Call 生态的融合**：OpenAI、Google 等巨头有自己的工具调用标准，生态统一仍需时间

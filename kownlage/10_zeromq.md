# 10. ZeroMQ 详解

## 一、概念：高性能异步消息库

### 1.1 什么是 ZeroMQ

ZeroMQ（也写作 0MQ、ZMQ）是一个高性能、异步的消息传递库。它不是传统意义上的消息队列（Message Queue），而是一个**嵌入式的网络通信库**。名字中的"Zero"代表：

- **Zero Broker**：不需要消息代理/中间件（与 RabbitMQ、Kafka 不同）
- **Zero Latency**：追求极低延迟
- **Zero Cost**：几乎零管理成本
- **Zero Administration**：无需部署额外的服务进程

ZeroMQ 的核心定位是：**像 Socket 一样易用，但提供了更高层次的消息传递抽象**。它把底层的 TCP/UDP/IPC/inproc 通信封装成统一的消息传递模式，开发者可以用极少的代码构建复杂的分布式通信架构。

### 1.2 核心特点

```
传统 Socket:      应用层 → Socket API → TCP/IP → 网络
ZeroMQ:          应用层 → ZeroMQ Pattern → ZeroMQ Engine → TCP/IPC/inproc → 网络/内存
```

- **消息导向**：传输的基本单位是消息（message），而不是字节流
- **异步 I/O**：底层使用异步 I/O 引擎，所有操作非阻塞
- **自动重连**：网络断开后自动尝试重连
- **多种传输协议**：tcp、ipc（进程间通信）、inproc（线程间通信）、pgm（多播）
- **多种消息模式**：内置多种经典分布式通信模式

### 1.3 Python 示例（pyzmq）

```python
import zmq

# 服务端
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    message = socket.recv()     # 接收请求
    socket.send(b"World")       # 发送回复

# 客户端
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

socket.send(b"Hello")          # 发送请求
message = socket.recv()        # 接收回复
```

---

## 二、核心模式

### 2.1 REQ-REP（请求-应答模式）

```
Client (REQ) ----request---->  Server (REP)
Client (REQ) <---reply------  Server (REP)
```

**特点：**
- 严格的一问一答：REQ 必须先 send 再 recv，REP 必须先 recv 再 send
- 同步通信模型，简单可靠
- 类似 HTTP 的请求/响应

**适用场景：**
- RPC 调用
- 简单的客户端-服务端通信
- 推理服务中客户端发送推理请求，服务端返回结果

**局限性：**
- 同步模型，性能受限
- 一个 REP 同时只能处理一个请求
- 任何一端崩溃都会导致对端阻塞

### 2.2 PUB-SUB（发布-订阅模式）

```
Publisher (PUB) ---broadcast---> Subscriber1 (SUB)
                ---broadcast---> Subscriber2 (SUB)
                ---broadcast---> Subscriber3 (SUB)
```

**特点：**
- 一对多广播：Publisher 不关心有多少 Subscriber
- Subscriber 通过 topic 过滤消息
- 单向通信：Publisher 只发不收，Subscriber 只收不发
- **有"慢加入"问题**：Subscriber 连接后可能错过之前的消息

**适用场景：**
- 日志广播
- 状态通知
- 推理框架中调度器向多个 Worker 广播全局配置/状态变更

```python
# Publisher
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5556")
pub.send_string("topic1 Hello subscribers")

# Subscriber
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:5556")
sub.setsockopt_string(zmq.SUBSCRIBE, "topic1")  # 只订阅 topic1
message = sub.recv_string()
```

### 2.3 PUSH-PULL（推-拉/管道模式）

```
Ventilator (PUSH) ---task1---> Worker1 (PULL)
                  ---task2---> Worker2 (PULL)  --result--> Sink (PULL)
                  ---task3---> Worker3 (PULL)
```

**特点：**
- 负载均衡：多个 PULL 端自动做 round-robin 负载均衡
- 单向通信：PUSH 只发，PULL 只收
- 流水线模型：适合多阶段数据处理

**适用场景：**
- 任务分发（类似工作队列）
- 推理框架中调度器将请求分发到多个 Worker
- MapReduce 风格的并行计算
- **vLLM/SGLang 中的调度器 → Worker 任务下发**

```python
# Ventilator (Task distributor)
push = context.socket(zmq.PUSH)
push.bind("tcp://*:5557")
for task in tasks:
    push.send_json(task)

# Worker
pull = context.socket(zmq.PULL)
pull.connect("tcp://localhost:5557")
while True:
    task = pull.recv_json()
    process(task)
```

### 2.4 ROUTER-DEALER（路由-经销商模式）

```
Client1 (REQ/DEALER) ─┐
Client2 (REQ/DEALER) ─┤─── ROUTER ─── DEALER ───┤─ Worker1 (REP/DEALER)
Client3 (REQ/DEALER) ─┘                          ├─ Worker2 (REP/DEALER)
                                                  └─ Worker3 (REP/DEALER)
```

**特点：**
- **ROUTER**：可以跟踪消息来源（通过 identity frame），实现定向回复
- **DEALER**：异步版的 REQ，可以同时发送多个请求，负载均衡分发
- 组合使用可构建异步请求-回复代理（Broker）

**ROUTER 的关键能力——消息身份追踪：**
```
ROUTER 收到消息格式: [identity][empty][data]
- identity: 发送者的身份标识，自动添加
- empty: 空帧，分隔符
- data: 实际数据
```

**适用场景：**
- 异步 RPC 代理
- 负载均衡代理
- 需要将回复精确路由到原始请求者的场景
- **推理框架中实现请求路由和结果返回**

```python
# Broker (ROUTER-DEALER proxy)
frontend = context.socket(zmq.ROUTER)  # 面向客户端
frontend.bind("tcp://*:5559")
backend = context.socket(zmq.DEALER)   # 面向 Worker
backend.bind("tcp://*:5560")
zmq.proxy(frontend, backend)  # 内置代理函数

# Worker
worker = context.socket(zmq.DEALER)
worker.connect("tcp://localhost:5560")
```

### 2.5 四种模式对比

| 特性 | REQ-REP | PUB-SUB | PUSH-PULL | ROUTER-DEALER |
|------|---------|---------|-----------|---------------|
| 通信方向 | 双向（一问一答）| 单向（广播）| 单向（流水线）| 双向（异步）|
| 负载均衡 | 无 | 无 | 自动 round-robin | 自动 |
| 消息丢失 | 不丢失 | 可能丢失 | 不丢失 | 不丢失 |
| 同步/异步 | 同步 | 异步 | 异步 | 异步 |
| 适用场景 | 简单 RPC | 广播通知 | 任务分发 | 复杂路由 |

---

## 三、与传统 Socket 的区别

### 3.1 无 Broker（无中间件）

```
传统消息队列（如 RabbitMQ/Kafka）:
  Producer → [Broker Server] → Consumer
  - 需要部署和维护 Broker 服务
  - Broker 是单点瓶颈

ZeroMQ:
  Producer → Consumer （直接通信）
  - 库级别集成，无额外进程
  - 消息路由逻辑内嵌在端点中
  - 更低延迟，更少运维负担
```

### 3.2 自动重连

传统 Socket：
- 连接断开后需要手动处理重连逻辑
- 需要自己实现心跳、超时、重试

ZeroMQ：
- **连接断开后自动重连**，对应用层透明
- **连接顺序无关**：bind 和 connect 的顺序可以任意，先 connect 后 bind 也可以
- **连接/断开事件自动处理**：消息在连接恢复后自动发送

```python
# 即使服务端还没启动，客户端也可以先 connect
# ZeroMQ 会在后台自动尝试连接
socket.connect("tcp://localhost:5555")
socket.send(b"Hello")  # 消息会缓存，等连接建立后自动发送
```

### 3.3 消息帧（Message Frame）

传统 Socket（TCP）：
- 字节流导向：TCP 是字节流协议，没有消息边界
- 需要自己处理粘包/拆包（如长度前缀、分隔符）
- 应用层需要自己定义协议格式

ZeroMQ：
- **消息导向**：每次 send/recv 是完整的消息，自动处理消息边界
- **多帧消息（Multipart Message）**：一个逻辑消息可以由多个帧组成
- **原子性发送**：多帧消息要么全部发送，要么全部不发送
- **零拷贝支持**：大消息可以避免不必要的内存拷贝

```python
# 多帧消息
socket.send_multipart([
    b"header",           # 第一帧：头部信息
    b"",                 # 第二帧：空帧（分隔符）
    b"actual data",      # 第三帧：数据
    large_tensor_bytes   # 第四帧：大数据（可零拷贝）
])

# 接收端完整接收
frames = socket.recv_multipart()
```

### 3.4 完整对比表

| 特性 | 传统 Socket (TCP) | ZeroMQ |
|------|-------------------|--------|
| 通信模型 | 字节流 | 消息 |
| 连接管理 | 手动 | 自动（重连、缓存）|
| 中间件 | N/A | 不需要 |
| 消息边界 | 无（需自行处理）| 自动 |
| 多对多通信 | 复杂 | 内置模式支持 |
| 线程安全 | 需自行处理 | Socket 不线程安全，但 Context 线程安全 |
| 传输协议 | TCP/UDP | tcp/ipc/inproc/pgm |
| 消息模式 | 无 | REQ-REP/PUB-SUB/PUSH-PULL 等 |
| 性能 | 取决于实现 | 高度优化，低延迟 |

---

## 四、在推理框架中的应用

### 4.1 vLLM 中的 ZeroMQ 应用

vLLM 使用 ZeroMQ 作为其**进程间通信（IPC）**的核心基础设施。

#### 4.1.1 架构概览

```
                    ┌─────────────────────────────────────┐
                    │           API Server                 │
                    │    (FastAPI/OpenAI Compatible)       │
                    └──────────────┬──────────────────────┘
                                   │ HTTP
                    ┌──────────────▼──────────────────────┐
                    │         LLM Engine                   │
                    │      (Scheduler / Coordinator)       │
                    └──────────────┬──────────────────────┘
                                   │ ZeroMQ (IPC/TCP)
                    ┌──────────────▼──────────────────────┐
                    │         Worker Process(es)           │
                    │   ┌─────────┐  ┌─────────┐          │
                    │   │ Worker0 │  │ Worker1 │  ...     │
                    │   │ (GPU 0) │  │ (GPU 1) │          │
                    │   └─────────┘  └─────────┘          │
                    └─────────────────────────────────────┘
```

#### 4.1.2 vLLM 中 ZeroMQ 的具体使用方式

1. **调度器（Scheduler）与 Worker 间通信**
   - 调度器通过 ZeroMQ 将调度决策（哪些请求要执行、batch 怎么组）发送给 Worker
   - Worker 执行完一步推理后，通过 ZeroMQ 返回结果
   - 使用 PUSH-PULL 或 ROUTER-DEALER 模式

2. **多进程部署模式**
   - 当使用 tensor parallelism 时，多个 GPU Worker 运行在不同进程中
   - 进程间通过 ZeroMQ 的 ipc:// 传输协议通信（共享内存，极低延迟）
   - 或通过 tcp:// 跨节点通信

3. **前端与引擎分离**
   - vLLM 支持将 API Server 和 Engine 部署在不同进程
   - 通过 ZeroMQ 的 REQ-REP 或 ROUTER-DEALER 模式连接

```python
# vLLM 中的典型 ZeroMQ 使用模式（简化示意）
# Engine 端（Publisher/Router）
self.input_socket = context.socket(zmq.PUSH)
self.input_socket.bind("ipc:///tmp/vllm_input")

self.output_socket = context.socket(zmq.PULL)
self.output_socket.bind("ipc:///tmp/vllm_output")

# Worker 端
self.input_socket = context.socket(zmq.PULL)
self.input_socket.connect("ipc:///tmp/vllm_input")

self.output_socket = context.socket(zmq.PUSH)
self.output_socket.connect("ipc:///tmp/vllm_output")
```

### 4.2 SGLang 中的 ZeroMQ 应用

SGLang 同样大量使用 ZeroMQ，其使用更为核心和深入。

#### 4.2.1 SGLang 的通信架构

```
HTTP Server (FastAPI)
       │
       │  ZeroMQ (ipc://)
       ▼
Tokenizer Manager ──ZeroMQ──> Scheduler ──ZeroMQ──> TP Workers
       │                          │
       │                    ┌─────┴─────┐
       │                    ▼           ▼
       │               Worker 0    Worker 1  ...
       │               (GPU 0)    (GPU 1)
       ▼
Detokenizer Manager
```

#### 4.2.2 SGLang 中的具体使用

1. **Tokenizer Manager ↔ Scheduler**
   - Tokenizer 进程通过 ZeroMQ 将 tokenize 后的请求发送给 Scheduler
   - Scheduler 通过 ZeroMQ 将完成的 token id 返回给 Detokenizer

2. **Scheduler ↔ Worker**
   - Scheduler 通过 ZeroMQ 控制 Worker 的执行
   - 发送 batch 信息、调度指令

3. **使用 ipc:// 协议**
   - SGLang 在单机场景大量使用 ipc:// 传输
   - 基于 Unix Domain Socket，延迟极低（微秒级）
   - 避免了 TCP 的协议开销

### 4.3 为什么推理框架选择 ZeroMQ 而非其他方案

| 方案 | 延迟 | 复杂度 | 适用性 |
|------|------|--------|--------|
| ZeroMQ | 极低（微秒级）| 低 | 非常适合 |
| gRPC | 较低（毫秒级）| 中 | 可用但有额外开销 |
| RabbitMQ/Kafka | 较高 | 高（需部署 Broker）| 不适合推理实时通信 |
| 共享内存 | 最低 | 高（需自行管理同步）| 可用但开发复杂 |
| Python multiprocessing.Queue | 中 | 低 | 性能不够 |

选择 ZeroMQ 的关键原因：
- **极低延迟**：推理服务对延迟极度敏感，ZeroMQ 的 ipc 传输可以达到微秒级
- **无 Broker**：不需要额外部署消息中间件，简化部署
- **跨语言**：底层 C++ 实现，Python 通过 pyzmq 绑定，性能接近原生
- **灵活的通信模式**：内置多种模式，适配推理框架中各种通信需求
- **自动重连**：Worker 崩溃重启后可自动恢复连接

---

## 五、性能特点

### 5.1 低延迟

- **ipc:// 传输**：基于 Unix Domain Socket，延迟通常在 **10-30 微秒**
- **inproc:// 传输**：线程间通信，延迟可达 **1-5 微秒**
- **tcp:// 传输**：本地回环约 **30-100 微秒**，跨网络取决于网络条件
- ZeroMQ 内部使用 **无锁队列（lock-free queue）** 减少同步开销

### 5.2 高吞吐

- 消息吞吐可达 **数百万消息/秒**（小消息场景）
- 底层 I/O 引擎经过高度优化
- 支持批量发送（batching），减少系统调用
- **零拷贝（zero-copy）**：大消息可以避免不必要的内存拷贝

### 5.3 智能缓冲

- 发送端和接收端都有内部缓冲区
- 当接收端处理慢时，消息在发送端缓冲（背压机制）
- 可配置高水位线（HWM, High Water Mark），防止内存溢出

```python
# 配置高水位线
socket.setsockopt(zmq.SNDHWM, 1000)  # 发送缓冲上限 1000 条
socket.setsockopt(zmq.RCVHWM, 1000)  # 接收缓冲上限 1000 条
```

### 5.4 推理场景中的性能考虑

在大模型推理中，ZeroMQ 传输的主要是：
- **调度指令**：很小的消息（几十到几百字节），延迟敏感
- **Token IDs**：中等大小的消息（几 KB），频率高
- **元数据**：请求参数、采样参数等

注意：**大规模张量数据（如 KV Cache、模型权重）通常不通过 ZeroMQ 传输**，而是使用：
- NCCL（GPU 间直接通信）
- 共享内存 + CUDA IPC
- RDMA（跨节点）

ZeroMQ 主要负责**控制面（Control Plane）**通信，而非**数据面（Data Plane）**的大规模数据传输。

---

## 六、面试要点总结

1. **ZeroMQ 是什么**：高性能异步消息库，不是消息队列，无需 Broker
2. **四种核心模式**：REQ-REP（同步请求应答）、PUB-SUB（广播）、PUSH-PULL（负载均衡分发）、ROUTER-DEALER（异步路由）
3. **与传统 Socket 的三大区别**：无 Broker、自动重连、消息帧（非字节流）
4. **推理框架中的角色**：控制面通信（调度器↔Worker、前端↔引擎），而非数据面大规模传输
5. **性能**：微秒级延迟、百万级吞吐、ipc 比 tcp 更快

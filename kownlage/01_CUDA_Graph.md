# 1. CUDA Graph 详解

---

## 一、什么是 CUDA Graph，核心原理

CUDA Graph 是 NVIDIA 在 CUDA 10 中引入的一项关键优化技术。它的核心思想是：将一系列 GPU 操作（kernel 启动、内存拷贝等）预先录制（capture）成一个"图"（Graph），然后在后续的执行中，只需要一次性提交整个图即可，而不需要逐个 kernel 地从 CPU 端发射。

要理解 CUDA Graph 的价值，首先要理解传统 GPU 执行模式的瓶颈。在传统模式下，每一个 GPU kernel 的启动都需要经历以下过程：

1. CPU 端准备 kernel 参数
2. CPU 通过 CUDA Driver API 向 GPU 发送启动命令
3. GPU 接收命令并执行 kernel
4. CPU 继续准备下一个 kernel

每次 kernel launch 都会产生大约 5-10 微秒的 CPU 端开销（launch overhead）。对于像大模型推理这样的场景，一次 forward pass 可能包含数百甚至上千个 kernel（矩阵乘法、LayerNorm、激活函数、Attention 等），累积的 launch overhead 可能达到毫秒级别，这在推理的 decode 阶段尤其明显——因为 decode 阶段每个 token 的 GPU 计算量很小（batch 较小时每个 kernel 执行时间可能只有几微秒），kernel launch overhead 占比可能超过 50%。

### CUDA Graph 的工作流程

分为三步：

**（1）录制（Capture）**：将一段 GPU 操作序列录制为一个图结构。在录制阶段，所有的 CUDA 操作不会真正执行，而是被记录到一个 `cudaGraph_t` 对象中。录制通常通过 `cudaStreamBeginCapture` 和 `cudaStreamEndCapture` 来完成。

**（2）实例化（Instantiate）**：将录制好的图编译为一个可执行的 `cudaGraphExec_t` 对象。在这个阶段，CUDA 运行时会对图进行优化，比如分析依赖关系、确定并行执行的可能性等。

**（3）重放（Launch/Replay）**：通过 `cudaGraphLaunch` 将整个图一次性提交到 GPU 执行。由于所有操作已经预先录制和优化，CPU 端只需要一次 launch 调用，极大减少了 CPU 端的开销。

### 核心原理的关键点

- **减少 CPU-GPU 交互次数**：传统方式 N 个 kernel 需要 N 次 launch，CUDA Graph 只需 1 次
- **减少 Driver 开销**：CUDA Driver 的内部状态检查、参数验证等操作在录制时只做一次
- **启用 GPU 端调度优化**：GPU 可以看到整个执行图，从而更好地安排执行顺序和资源分配
- **内存地址固定**：CUDA Graph 录制时会固定所有 tensor 的地址，后续重放时使用相同的地址，避免了每次重新分配和绑定

---

## 二、在推理框架中如何使用

### 2.1 vLLM 中的 CUDA Graph

vLLM 在 decode 阶段大量使用 CUDA Graph。具体实现方式：

- vLLM 会针对不同的 batch size 预先录制多个 CUDA Graph。例如，分别录制 batch_size=1,2,4,8,16,32... 的 graph
- 当实际请求到达时，vLLM 会选择大于等于实际 batch size 的最小预录制 graph 来执行，多余的位置用 padding 填充
- 由于 CUDA Graph 要求固定的 tensor 地址和形状，vLLM 使用固定大小的缓冲区（placeholder buffer），在 graph replay 之前通过 `cudaGraphExecKernelNodeSetParams` 或直接内存拷贝来更新输入数据
- vLLM 的 `CUDAGraphRunner` 类负责管理 graph 的录制、缓存和重放

### 2.2 SGLang 中的 CUDA Graph

SGLang 同样在 decode 阶段使用 CUDA Graph，并且做了进一步优化：

- SGLang 使用了更细粒度的 CUDA Graph 管理策略，支持更灵活的 batch size 范围
- SGLang 将 CUDA Graph 与 RadixAttention 结合，在 prefix caching 场景下也能使用 CUDA Graph
- SGLang 还支持 CUDA Graph 的动态 shape 特性（在较新版本的 CUDA 中），减少 padding 浪费

### 2.3 通用使用模式

- **Prefill 阶段通常不使用 CUDA Graph**，因为 prefill 的 sequence length 变化大，录制的 graph 数量会爆炸，且 prefill 阶段 GPU 计算密集，kernel launch overhead 占比很小
- **Decode 阶段是 CUDA Graph 的主要应用场景**，因为每次只生成一个 token，计算量小，kernel launch overhead 占比大
- 一般框架会在启动（warmup）阶段预先录制常用 batch size 的 graph，避免运行时的录制开销

---

## 三、优缺点与适用场景

### 优点

- 显著降低 kernel launch overhead，decode 阶段性能提升通常在 10%-30%，小 batch 场景下提升可达 50% 以上
- 减少 CPU 占用，使得 CPU 可以用于其他任务（如调度、采样）
- GPU 端可以更好地流水线化执行，提高 GPU 利用率

### 缺点

- **要求固定的计算图**：tensor shape、kernel 参数等必须在录制时确定。这意味着对于 dynamic shape 场景需要额外处理（padding 或多 graph 缓存）
- **内存开销**：每个录制的 graph 都会占用额外的 GPU 显存（用于存储中间 tensor 和执行计划），多个 batch size 的 graph 缓存会消耗可观的显存
- **录制开销**：首次录制需要时间，且在框架 warmup 阶段会增加启动延迟
- **不支持动态控制流**：if-else、while 等动态逻辑无法在 graph 中录制（CUDA 12 的条件节点提供了有限支持）
- **调试困难**：graph 中的错误定位比较困难，不如逐 kernel 执行直观

### 适用场景

- 大模型推理的 decode 阶段（最典型的应用场景）
- 小 batch、多 kernel 的推理场景
- 计算图固定、输入 shape 可枚举的场景
- 对延迟（latency）要求极高的在线服务

### 不适用场景

- Prefill 阶段（sequence length 变化大）
- 训练过程（梯度计算图可能变化）
- 输入 shape 高度动态且不可枚举的场景

---

## 四、与 CUDA Stream 的关系

CUDA Stream 和 CUDA Graph 是两个不同层次的概念，但紧密相关：

CUDA Stream 是 GPU 命令的执行队列。同一个 stream 中的操作按顺序执行，不同 stream 之间的操作可以并行。Stream 是 GPU 并发执行的基本机制。每个 kernel launch 都必须指定一个 stream。

它们的关系：

1. **CUDA Graph 的录制是在特定 stream 上完成的**。通过 `cudaStreamBeginCapture(stream)` 开始录制，该 stream 上后续的所有操作都会被捕获到 graph 中
2. **CUDA Graph 的重放（launch）也是在某个 stream 上执行的**。`cudaGraphLaunch(graphExec, stream)` 将整个 graph 提交到指定的 stream
3. **CUDA Graph 内部可以包含多个 stream 的操作**。在录制时如果涉及多个 stream 之间的事件同步，graph 会正确捕获这些依赖关系，形成一个 DAG（有向无环图）
4. **传统的 stream-based 执行是"逐条命令提交"，而 CUDA Graph 是"批量命令提交"**。可以类比为：stream 是"每次寄一封信"，graph 是"把所有信打包一次寄出"
5. **在推理框架中**，通常的做法是：在主 stream 上录制 graph，然后在主 stream 上重放 graph。如果需要 CPU-GPU overlap，可能会使用额外的 stream 来并行执行其他操作（如 KV cache 管理、采样等），而把主要的模型前向计算放在 CUDA Graph 中

**总结**：CUDA Stream 负责命令队列的管理和并发控制，CUDA Graph 负责将一系列命令打包以减少提交开销。两者配合使用可以同时获得并发执行和低 launch overhead 的收益。

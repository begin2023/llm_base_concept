# 23. 分布式推理并行策略

---

## 一、为什么需要分布式推理

单个 GPU 的显存有限（A100 80GB，H100 80GB），大型 LLM 的参数量往往远超单卡显存：

- Llama-3 70B（BF16）：~140 GB → 需要至少 2 张 H100
- Llama-3 405B（BF16）：~810 GB → 需要至少 10 张 H100
- DeepSeek V3 671B（BF16）：~1340 GB → 需要至少 17 张 H100

因此需要将模型分布到多个 GPU 上，通过不同的并行策略协同推理。

---

## 二、主要并行策略

### 2.1 Tensor Parallelism（张量并行，TP）

**原理**：将单个矩阵/张量沿某个维度切分到多个 GPU，每个 GPU 计算部分结果，最后通过 AllReduce 合并。

**具体切分方式**（以 Linear 层为例，输入 X，权重 W）：

**按列切分（Column Parallel Linear）**：
```
W = [W_0 | W_1 | W_2 | W_3]  → 每个 GPU 有 W/4 列
每个 GPU: Y_i = X × W_i
输出: Y = [Y_0, Y_1, Y_2, Y_3]（在 GPU 间分布）
```

**按行切分（Row Parallel Linear）**：
```
W = [W_0; W_1; W_2; W_3]  → 每个 GPU 有 W/4 行
每个 GPU: Y_i = X_i × W_i （X 也需要按列切分）
输出: Y = AllReduce(Y_0 + Y_1 + Y_2 + Y_3)
```

**Transformer 中的 TP**（Megatron-LM 方式）：

```
Attention 层:
  Q, K, V 矩阵：按 head 维度切分（每个 GPU 负责若干 head）
  Output 矩阵：按行切分
  AllReduce：Attention 输出后 AllReduce

FFN 层:
  W_up（或 W_gate）：按列切分
  W_down：按行切分
  AllReduce：FFN 输出后 AllReduce
```

**通信模式**：每个 Transformer 层需要 2 次 AllReduce（Attention 后 + FFN 后）

**适用场景**：
- 同一节点内的多 GPU（NVLink 带宽高，AllReduce 延迟低）
- TP 度通常为 2、4、8（受 AllReduce 延迟限制，不适合跨节点）

**代码示例（HuggingFace + DeepSpeed TP）**：
```python
import deepspeed

model = AutoModelForCausalLM.from_pretrained("...")
# 使用 DeepSpeed 的 TP
ds_engine = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 4},
    dtype=torch.bfloat16,
)
```

### 2.2 Pipeline Parallelism（流水线并行，PP）

**原理**：将 Transformer 的不同层分配到不同 GPU，输入数据像流水线一样依次经过各 GPU。

**示例**（32 层模型，4 卡 PP）：
```
GPU 0: Layer 0-7   (Stage 0)
GPU 1: Layer 8-15  (Stage 1)
GPU 2: Layer 16-23 (Stage 2)
GPU 3: Layer 24-31 (Stage 3)

数据流：输入 → GPU0 → GPU1 → GPU2 → GPU3 → 输出
```

**朴素 PP 的问题：GPU 利用率低（Bubble）**
```
时间轴：
GPU 0: [计算 micro-batch 1][等待][等待][等待]
GPU 1: [等待][计算 micro-batch 1][等待][等待]
GPU 2: [等待][等待][计算 micro-batch 1][等待]
GPU 3: [等待][等待][等待][计算 micro-batch 1]
```
每个 GPU 大部分时间在等待（"PP Bubble"），GPU 利用率很低。

**解决方案：Micro-batching 流水线**
```
时间轴（4个 micro-batch）：
GPU 0: [mb1][mb2][mb3][mb4][backward...]
GPU 1:     [mb1][mb2][mb3][mb4][backward...]
GPU 2:         [mb1][mb2][mb3][mb4][backward...]
GPU 3:             [mb1][mb2][mb3][mb4][backward...]
```
通过将 batch 切分为 micro-batch，减少 bubble。

**推理中的 PP**：
- 推理中没有 backward，PP bubble 问题依然存在
- 通常只在模型太大无法放入 TP 范围时才使用 PP
- PP 通信量小（只传输 hidden state），适合跨节点

### 2.3 Data Parallelism（数据并行，DP）

**原理**：每个 GPU 有完整的模型副本，不同 GPU 处理不同的请求（batch）。

**在推理中的 DP**：
- 对于能放入单卡的小模型，直接多开实例即可
- 配合负载均衡器（如 Nginx、HAProxy），将请求分发到不同实例
- 无需 GPU 间通信，是最简单高效的扩展方式

### 2.4 Expert Parallelism（专家并行，EP）

专用于 MoE 模型：

**原理**：将不同的 Expert 分配到不同 GPU，每个 token 通过 Gate 路由到对应 GPU 的 Expert 计算。

```
GPU 0: Expert 0, Expert 1, Expert 2, Expert 3
GPU 1: Expert 4, Expert 5, Expert 6, Expert 7
GPU 2: Expert 8, Expert 9, Expert 10, Expert 11
GPU 3: Expert 12, Expert 13, Expert 14, Expert 15

数据流：
  1. Gate 计算每个 token 应该去哪个 Expert
  2. All-to-All：每个 token 发送到对应的 Expert 所在 GPU
  3. Expert 计算
  4. All-to-All：结果返回原 GPU
```

**通信开销**：EP 需要 2 次 All-to-All 操作，代价较高但通信量与 token 数成正比，不随 EP 度增大。

### 2.5 Sequence Parallelism（序列并行，SP）

在长序列推理（Ring Attention）场景中：

**原理**：将输入序列按 token 维度切分到不同 GPU，每个 GPU 只处理部分序列的 attention。

```
序列长度 = 128K tokens，4 卡 SP：
GPU 0: token 0-32K
GPU 1: token 32K-64K
GPU 2: token 64K-96K
GPU 3: token 96K-128K
```

由于 Attention 需要 token 之间的交互（Query 要与所有 Key 做 dot product），SP 需要在 GPU 间传递 Key/Value 信息（Ring Attention 方式）。

---

## 三、并行策略组合（3D 并行 / 4D 并行）

大规模部署时，通常组合多种并行策略：

### 3.1 TP + PP

```
总 GPU 数 = TP × PP

例如：8 卡，TP=4，PP=2
  Node 0: GPU 0,1,2,3 → TP=4（处理前一半 layers）
  Node 1: GPU 4,5,6,7 → TP=4（处理后一半 layers）
```

### 3.2 TP + EP（MoE 模型）

```
例如：DeepSeek V3 在 H100 集群上：
  TP=8（单节点内 8 张 H100）
  EP=32（32 个节点，每节点负责部分 Expert）

  Attention：TP 并行（节点内 NVLink）
  Expert：EP 并行（跨节点 InfiniBand）
```

### 3.3 TP + DP

```
小批量：使用 TP 切分模型到多卡
大批量：多个 TP 组构成 DP（模型复制）
```

---

## 四、通信拓扑与硬件

### 4.1 节点内通信

- **NVLink**：GPU 间直接通信，带宽极高（H100：900 GB/s bidirectional）
- **PCIe**：通过 CPU，带宽较低（PCIe 5.0 x16：~64 GB/s）
- 建议：TP 在节点内（利用 NVLink），PP 跨节点

### 4.2 节点间通信

- **InfiniBand（IB）**：高带宽低延迟（HDR：200 Gbps，NDR：400 Gbps）
- **RoCE**：基于以太网的 RDMA，成本低于 IB
- **以太网（TCP/IP）**：延迟高，仅用于控制面

### 4.3 通信操作

| 操作 | 用途 | 参与 GPU |
|------|------|---------|
| AllReduce | TP 中合并梯度/激活 | 同 TP 组内所有 GPU |
| All-to-All | EP 中路由 token 到 Expert | 同 EP 组内所有 GPU |
| P2P（Send/Recv） | PP 中传递 hidden state | 相邻 Pipeline Stage |
| AllGather | SP 中收集完整序列 | 同 SP 组内所有 GPU |
| ReduceScatter | SP 中分散结果 | 同 SP 组内所有 GPU |

---

## 五、vLLM 和 SGLang 中的并行策略

### 5.1 vLLM

```python
# Tensor Parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,   # TP=4
    pipeline_parallel_size=2,  # PP=2（实验性）
)

# 命令行启动
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

vLLM 使用 Ray 进行多 GPU 协调，每个 GPU 运行一个 Worker 进程。

### 5.2 SGLang

```bash
# SGLang TP
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp 4

# SGLang MoE EP + TP
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --ep 8
```

---

## 六、延迟 vs 吞吐量的权衡

| 并行策略 | 延迟影响 | 吞吐量影响 | 主要通信 |
|---------|--------|----------|---------|
| TP | 增加（AllReduce 延迟） | 提升（并行计算） | AllReduce |
| PP | 增加（Pipeline Bubble） | 中性/略提升 | P2P |
| DP | 无影响 | 线性提升 | 无（无状态共享） |
| EP | 增加（All-to-All） | 提升（Expert 并行） | All-to-All |
| SP | 增加（Ring 通信） | 支持更长序列 | AllGather/ReduceScatter |

---

## 七、总结

分布式推理并行策略的选择原则：

1. **TP 优先**：最常用，适合同节点多 GPU，NVLink 下通信开销小
2. **PP 作为补充**：当 TP 已达单节点 GPU 上限，通过 PP 跨节点扩展
3. **EP 用于 MoE**：Expert 太多时用 EP 分散 Expert
4. **DP 用于扩容**：满足更高 QPS，横向复制整个 TP/PP 组
5. **SP 用于超长序列**：序列太长（> 32K）单卡 KV Cache 不够时

合理组合以上策略，是大模型推理系统工程设计的核心挑战之一。

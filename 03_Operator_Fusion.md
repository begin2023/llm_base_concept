# 3. 算子融合（Operator Fusion）详解

---

## 一、概念与原理

算子融合（Operator Fusion，也叫 Kernel Fusion）是一种将多个独立的计算操作合并为单个 GPU kernel 执行的优化技术。它是大模型推理优化中最基础、也最重要的技术之一。

### 1.1 Memory Wall 问题

要理解算子融合的原理，首先要理解 GPU 计算中的"Memory Wall"（内存墙）问题：

- GPU 的计算能力（如 A100 的 312 TFLOPS FP16）远远超过其内存带宽（如 A100 的 2 TB/s）
- 很多操作是"内存带宽受限"（Memory-Bound）的，也就是说 GPU 大部分时间在等待数据从显存搬运到计算单元，而不是在做计算
- 典型的 memory-bound 操作包括：逐元素操作（ReLU、GELU）、归一化（LayerNorm、RMSNorm）、残差连接（Add）等

### 1.2 未融合 vs 融合

在没有算子融合的情况下，每个操作是一个独立的 kernel：

1. Kernel A：从显存读取输入 -> 计算 -> 将结果写回显存
2. Kernel B：从显存读取 Kernel A 的输出 -> 计算 -> 将结果写回显存
3. Kernel C：从显存读取 Kernel B 的输出 -> 计算 -> 将结果写回显存

这里的问题是：每个 kernel 之间都有一次"写回显存 + 重新读取"的过程（称为 memory round-trip），而这些中间结果本可以直接在寄存器或共享内存中传递。

算子融合将多个操作合并为一个 kernel：

> 融合 Kernel ABC：从显存读取输入 -> 计算 A -> 在寄存器中直接传递 -> 计算 B -> 在寄存器中直接传递 -> 计算 C -> 将最终结果写回显存

### 1.3 融合带来的收益

- **减少显存读写次数**：中间结果不需要写回显存，显著降低内存带宽消耗
- **减少 kernel launch 次数**：多个 kernel 合并为一个，降低 launch overhead
- **提高数据局部性**：数据在寄存器和共享内存中复用，利用 GPU 的存储层次结构
- **减少显存占用**：中间结果不需要在显存中分配空间

### 1.4 算子融合的分类

1. **横向融合（Horizontal Fusion）**：将多个独立的、可并行的操作融合。例如，将 Q、K、V 三个独立的线性变换融合为一个大的矩阵乘法
2. **纵向融合（Vertical Fusion）**：将串行依赖的操作融合。例如，将 MatMul -> BiasAdd -> Activation 三步融合为一个 kernel
3. **混合融合**：同时进行横向和纵向融合

---

## 二、常见融合模式

### 2.1 QKV 融合（QKV Projection Fusion）

- **原始操作**：三次独立的矩阵乘法 `Q = X @ Wq, K = X @ Wk, V = X @ Wv`
- **融合后**：将 Wq、Wk、Wv 拼接为一个大矩阵 W_qkv，执行一次矩阵乘法 `QKV = X @ W_qkv`，然后 split 得到 Q、K、V
- **收益**：减少了两次矩阵乘法的 kernel launch 和内存读写，一次大矩阵乘法的 GPU 利用率也高于三次小矩阵乘法（更好的并行度和缓存利用）
- 这是一种横向融合的典型例子
- 在 GQA（Grouped Query Attention）中也适用，只是 K 和 V 的维度更小

### 2.2 Add + LayerNorm / Add + RMSNorm 融合

- **原始操作**：(a) `residual = x + attention_output`  (b) `output = LayerNorm(residual)`
- 这两个操作都是逐元素操作，都是 memory-bound 的
- **融合后**：一个 kernel 中完成残差连接和归一化，只读一次输入数据，写一次输出数据
- 进一步扩展：有的框架会做 Add + LayerNorm + 下一层的线性变换 的部分融合
- RMSNorm 比 LayerNorm 更容易融合，因为 RMSNorm 不需要计算均值，只计算方差，计算更简单
- 在 Transformer 的每一层中，这个融合模式会出现两次（attention 后和 FFN 后），收益显著

### 2.3 GEMM + BiasAdd + Activation 融合

- **原始操作**：(a) `y = x @ W`  (b) `y = y + bias`  (c) `y = activation(y)`，其中 activation 可以是 ReLU、GELU、SiLU/Swish 等
- **融合后**：在矩阵乘法的 epilogue（收尾阶段）中直接完成 bias 加法和激活函数计算
- **实现方式**：利用 CUTLASS 或 cuBLAS 的 epilogue fusion 机制，在矩阵乘法的最后一步（将结果从寄存器写回显存时）插入自定义计算
- **收益**：避免了 bias 和 activation 各自独立 kernel 带来的两次额外显存读写

### 2.4 SwiGLU / GeGLU 融合（用于 FFN）

- **原始操作**：`gate = x @ W_gate, up = x @ W_up, output = SiLU(gate) * up`
- **融合策略**：先做 Gate + Up 的矩阵乘法融合（类似 QKV 融合），然后将 SiLU 激活和逐元素乘法融合为一个 kernel
- 在 LLaMA 系列模型中非常常见

### 2.5 Rotary Position Embedding (RoPE) 融合

- 将 RoPE 的位置编码计算与 Q、K 的变换融合
- 在 QKV 投影的 epilogue 中直接应用旋转变换
- 避免单独为 RoPE 分配中间 tensor 和启动 kernel

### 2.6 Softmax + Mask 融合

- 在 Attention 中将 mask 的加法和 softmax 计算融合
- FlashAttention 更进一步，将整个 `Q@K^T -> Mask -> Softmax -> @V` 全部融合

### 2.7 AllReduce 融合（分布式场景）

- 将 AllReduce 通信与之前的计算或之后的计算融合
- 例如 GEMM + AllReduce + BiasAdd 融合，在 AllReduce 的同时完成 bias 加法

---

## 三、在推理框架中的实现

### 3.1 TensorRT / TensorRT-LLM

- TensorRT 是 NVIDIA 的推理优化引擎，内置了强大的自动算子融合能力
- TensorRT 的优化器（builder）会自动识别可融合的模式，包括上述所有常见模式
- TensorRT 使用 NVIDIA 内部的高度优化 kernel 库，包括 cuBLAS、cuDNN 和自研的融合 kernel
- TensorRT-LLM 在 TensorRT 的基础上，针对 LLM 推理做了大量专用的融合 kernel：
  - Fused Multi-Head Attention / Grouped Query Attention kernel
  - Fused QKV GEMM kernel
  - Fused Gate + Up + SiLU kernel
  - Fused Add + RMSNorm kernel
  - Fused RoPE kernel
- TensorRT 的融合策略是在图优化阶段（Graph Optimization）完成的，使用 pattern matching 来识别可融合的子图

### 3.2 自定义 CUDA Kernel

- 很多推理框架直接手写 CUDA kernel 来实现融合
- **优点**：可以针对特定模型架构做极致优化，灵活性最高
- **缺点**：开发成本高，需要深厚的 CUDA 编程经验
- 典型例子：
  - **FlashAttention**：手写 CUDA kernel 将整个 attention 的 QK^T -> scale -> mask -> softmax -> AV 全部融合，并使用 tiling 技术解决显存问题
  - **FlashInfer**：提供了丰富的融合 attention kernel，支持 PagedAttention、各种 attention 变体
  - **vLLM 的自定义 kernel**：包含融合的 RoPE、fused add-rmsnorm、fused moe 等
  - **SGLang** 同样集成了大量自定义融合 kernel

### 3.3 Triton（OpenAI）

- Triton 是 OpenAI 开发的一种 GPU 编程语言，比 CUDA 更高层，编译器会自动做一些融合优化
- 很多推理框架使用 Triton 来快速开发融合 kernel，如 fused softmax、fused layernorm 等
- Triton 的好处是开发效率高，编译器可以自动处理 tiling、共享内存管理等底层细节
- 但 Triton 生成的 kernel 性能通常略低于手写 CUDA kernel（约 90%-95% 的性能）

### 3.4 CUTLASS（NVIDIA）

- CUTLASS 是 NVIDIA 的模板化 CUDA 库，提供了高性能的矩阵乘法实现
- CUTLASS 的 epilogue 机制允许用户在矩阵乘法的最后阶段插入自定义的逐元素操作（bias、activation 等），实现 GEMM + 后处理的融合
- TensorRT 和很多推理框架内部都使用 CUTLASS 来实现融合的 GEMM kernel

### 3.5 torch.compile / Inductor

- PyTorch 2.0 的 torch.compile 通过 Inductor 后端可以自动进行算子融合
- Inductor 会将多个逐元素操作融合为一个 Triton kernel
- 对于简单的融合模式（如 bias + activation）效果不错，但对复杂的融合（如 FlashAttention 级别）仍需手动实现

---

## 四、对显存和计算效率的影响

### 4.1 对显存的影响

- **减少中间 tensor 的显存占用**：融合后的 kernel 不需要在显存中存储中间结果。例如，在未融合的 Attention 中，`Q@K^T` 会产生一个 `[batch, heads, seq_len, seq_len]` 的注意力矩阵，对长序列来说这个矩阵非常大（seq_len=4096 时约需数 GB）。FlashAttention 通过融合完全消除了这个中间矩阵
- 典型数据：
  - Add + RMSNorm 融合：节省一个 hidden_size 大小的中间 tensor
  - QKV 融合：节省两个中间矩阵的显存
  - FlashAttention：将注意力计算的额外显存从 O(N^2) 降低到 O(N)，这是质的飞跃
- 显存的节省意味着可以支持更大的 batch size 或更长的序列，间接提升吞吐

### 4.2 对计算效率的影响

- **对 memory-bound 操作的提升最大**：
  - 逐元素操作融合（如 Add + RMSNorm）：性能提升可达 2-3 倍，因为大幅减少了显存读写次数
  - 激活函数融合（如 GEMM + GELU）：GEMM 本身是 compute-bound 的，但 epilogue fusion 可以"免费"获得激活函数计算，相当于一定的性能提升
- **对 compute-bound 操作的提升主要来自减少 kernel launch overhead**：
  - QKV 融合：一次大 GEMM 替代三次小 GEMM，减少了 launch overhead 和 GPU 调度开销，同时大 GEMM 通常有更好的 SM 利用率
- **整体效果**：
  - 算子融合通常可以将 Transformer 推理的端到端延迟降低 30%-50%
  - 对于 decode 阶段（小 batch，多 memory-bound 操作），融合的收益更大
  - 对于 prefill 阶段（大序列，GEMM 主导），融合的收益主要体现在减少中间显存占用和 FlashAttention 类的优化

### 4.3 融合的局限性

- 不是所有操作都能融合：跨越同步屏障的操作、涉及不同并行策略的操作难以融合
- 融合 kernel 的开发和调试成本高
- 过度融合可能导致单个 kernel 过于复杂，寄存器压力增大，反而降低性能（寄存器溢出到 local memory 会大幅降低性能）
- 需要在融合粒度和 kernel 复杂度之间找到平衡点
- 不同 GPU 架构（Ampere vs Hopper vs Blackwell）可能需要不同的融合策略

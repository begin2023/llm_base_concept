# LLM Base Concept

LLM 推理与部署核心概念系列文档，覆盖从底层硬件优化到上层工程实践的 38 个主题。

## 目录

### 硬件与底层优化
| # | 主题 |
|---|------|
| 01 | [CUDA Graph](01_CUDA_Graph.md) |
| 02 | [CPU Overlap](02_CPU_Overlap.md) |
| 03 | [算子融合（Operator Fusion）](03_Operator_Fusion.md) |
| 12 | [RDMA](12_rdma.md) |

### 注意力机制
| # | 主题 |
|---|------|
| 04 | [FlashAttention & FlashInfer](04_FlashAttention_FlashInfer.md) |
| 26 | [MHA / GQA / MQA / MLA / Cross Attention / DSA / NSA](26_Attention_Variants_MHA_GQA_MQA_MLA_CrossAttn_DSA_NSA.md) |
| 27 | [位置编码：RoPE & ALiBi](27_position_encoding_rope_alibi.md) |
| 33 | [RoPE Scaling / NTK-Aware 长度外推](33_rope_scaling_ntk.md) |

### 推理引擎核心
| # | 主题 |
|---|------|
| 11 | [PagedAttention](11_paged_attention.md) |
| 13 | [Continuous Batching](13_continuous_batching.md) |
| 14 | [KV Cache 管理](14_kv_cache.md) |
| 28 | [Prefix Caching / Radix Attention](28_prefix_caching_radix_attention.md) |
| 29 | [Chunked Prefill](29_chunked_prefill.md) |
| 19 | [投机解码 / MTP](19_speculative_decoding_mtp.md) |
| 25 | [推理续写](25_inference_continuation.md) |

### 模型架构与量化
| # | 主题 |
|---|------|
| 05 | [MoE（Mixture of Experts）](05_MoE_Mixture_of_Experts.md) |
| 15 | [量化：AWQ / GPTQ / FP8](15_quantization_awq_gptq_fp8.md) |
| 24 | [每一个权重的含义](24_weight_meanings.md) |
| 35 | [Tokenizer：BPE & SentencePiece](35_tokenizer_bpe_sentencepiece.md) |
| 37 | [LoRA / QLoRA 推理适配](37_lora_qlora_inference.md) |

### 分布式与部署
| # | 主题 |
|---|------|
| 10 | [ZeroMQ](10_zeromq.md) |
| 16 | [模型加载加速](16_model_loading_acceleration.md) |
| 20 | [PD 分离](20_pd_disaggregation.md) |
| 21 | [AF 分离](21_af_disaggregation.md) |
| 23 | [分布式并行策略](23_distributed_parallelism.md) |
| 31 | [长上下文推理：Ring Attention & 序列并行](31_long_context_ring_attention.md) |
| 38 | [显存估算与模型部署规划](38_vram_estimation_deployment.md) |

### 解码与采样
| # | 主题 |
|---|------|
| 17 | [结构化输出](17_structured_output.md) |
| 34 | [Guided Decoding / Constrained Decoding](34_guided_decoding.md) |
| 36 | [Sampling 策略：Temperature / Top-K / Top-P / Min-P](36_sampling_strategies.md) |

### 多模态
| # | 主题 |
|---|------|
| 32 | [多模态推理：Vision Encoder + LLM](32_multimodal_inference.md) |

### 工程实践
| # | 主题 |
|---|------|
| 06 | [Function Call & Agent](06_function_call_agent.md) |
| 07 | [MCP（Model Context Protocol）](07_mcp.md) |
| 08 | [Context Engineering](08_context_engineering.md) |
| 09 | [Prompt Engineering](09_prompt_engineering.md) |
| 18 | [vLLM vs SGLang](18_vllm_vs_sglang.md) |
| 22 | [可观测性与性能分析](22_observability_profiling.md) |
| 30 | [RLHF / DPO / GRPO](30_rlhf_dpo_grpo.md) |

# 35. Tokenizer（BPE、SentencePiece）

---

## 一、为什么需要 Tokenizer

LLM 以 token 为基本单位处理文本，而不是字符或单词。Tokenizer 负责：

1. **Encoding**：将原始文本字符串转换为 token ID 序列
2. **Decoding**：将 token ID 序列转换回文本字符串

选择合适的分词策略至关重要：
- **字符级**（Character-level）：词表小（约 256），但序列极长，计算开销大
- **单词级**（Word-level）：词表大（>100K），但无法处理 OOV（未知词）和形态变化
- **子词级（Subword）**：介于两者之间，词表适中（32K-150K），兼顾两者优势

现代 LLM 几乎都使用子词分词。

---

## 二、BPE（Byte Pair Encoding，字节对编码）

BPE 是目前最主流的子词分词算法，被 GPT-2、GPT-4、Llama 等模型使用。

### 2.1 BPE 的训练过程

**初始化**：将文本按字符分割（或按 UTF-8 字节），建立初始词表。

**迭代合并**：
1. 统计所有相邻 token 对（bigram）的频率
2. 选择最高频的 token 对，将其合并为新 token，加入词表
3. 更新文本中所有该 token 对为新 token
4. 重复直到词表大小达到目标

**示例**：

```
初始文本（字符级）：
"low", "lower", "newest", "widest"
→ l o w, l o w e r, n e w e s t, w i d e s t

初始 token 对频率：
(e, s): 3
(s, t): 3
(e, r): 1
...

Step 1: 合并 (e, s) → es
  词表新增: "es"
  文本变为: l o w, l o w e r, n e w es t, w i d es t

Step 2: 合并 (es, t) → est
  ...

最终词表包含: "lo", "low", "es", "est", "newest", ...
```

### 2.2 Byte-level BPE（GPT-2 / GPT-4 / Llama 使用）

**问题**：原始 BPE 对 Unicode 字符（如中文、日文）处理不好，可能把多字节字符拆成不完整的字节。

**解决**：Byte-level BPE 将文本先编码为 UTF-8 字节（0-255），再在字节上做 BPE：
- 初始词表：256 个字节（0x00-0xFF）
- 所有文本都能无损编码，没有 OOV 问题
- GPT-2 的 `tiktoken` 和 Llama 的 tokenizer 都使用这种方式

### 2.3 tiktoken（OpenAI 的 BPE 实现）

tiktoken 是 OpenAI 开源的高效 BPE 实现：

```python
import tiktoken

# GPT-4 的 tokenizer（cl100k_base）
enc = tiktoken.get_encoding("cl100k_base")

# 编码
tokens = enc.encode("Hello, World! 你好世界")
print(tokens)  # [9906, 11, 4435, 0, 220, 57668, 53901, 16界]

# 解码
text = enc.decode(tokens)
print(text)  # "Hello, World! 你好世界"

# 词表大小
print(enc.n_vocab)  # 100277

# Token 数量（估算成本）
print(len(tokens))
```

---

## 三、SentencePiece

SentencePiece 是 Google 开发的分词库，被 T5、BERT（multilingual）、Llama（内部实现用 SP）等使用。

### 3.1 SentencePiece 的特点

**语言无关性**：
- 直接在原始 Unicode 字符上操作（不需要预先分词）
- 天然支持多语言（中文、日文等无空格分隔的语言）
- 将空格作为特殊字符处理（`▁` 表示词首空格）

**两种分词算法**：

1. **BPE**：与上述 BPE 相同，只是实现框架不同
2. **Unigram Language Model（Unigram LM）**：
   - 不同于 BPE 的自底向上合并，Unigram LM 是自顶向下的
   - 初始化一个大词表，然后迭代删除对整体似然贡献最小的 token
   - 最终保留目标大小的词表
   - T5、ALBERT 等使用 Unigram LM

### 3.2 SentencePiece 的使用

```python
import sentencepiece as spm

# 训练（通常模型已经包含训练好的 tokenizer）
# spm.SentencePieceTrainer.train(...)

# 加载 Llama 的 tokenizer
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")

# 编码
tokens = sp.Encode("Hello, World! 你好世界")
print(tokens)  # [1, 15043, 29892, 2787, 29991, 29871, 20025, ...]

# 解码
text = sp.Decode(tokens)

# 词表大小
print(sp.vocab_size())  # 32000（Llama-2）
```

### 3.3 HuggingFace Tokenizers

HuggingFace 的 `tokenizers` 库（Rust 实现，速度快）统一了各种分词算法：

```python
from transformers import AutoTokenizer

# 加载任意模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 编码
encoding = tokenizer("Hello, World!", return_tensors="pt")
input_ids = encoding["input_ids"]  # tensor([[1, 15043, 29892, 2787, 29991]])

# 批量编码（自动 padding）
batch_encoding = tokenizer(
    ["Hello", "Hello, World!"],
    padding=True,
    return_tensors="pt"
)

# 解码
text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

---

## 四、Special Tokens（特殊 Token）

### 4.1 常见特殊 Token

| Token | 含义 | 示例 |
|-------|------|------|
| `<s>` / `<bos>` | 序列开始（Beginning of Sequence） | Llama 系列 |
| `</s>` / `<eos>` | 序列结束（End of Sequence） | 所有模型 |
| `<pad>` | 填充（Padding），用于对齐 batch | BERT, T5 |
| `<unk>` | 未知 token（OOV） | Byte-level BPE 不需要 |
| `<mask>` | 被遮蔽的 token（用于 MLM 预训练） | BERT |

### 4.2 Chat Template 特殊 Token

现代 instruction-tuned 模型有特殊的对话格式 token：

**Llama-2 Chat 格式**：
```
<s>[INST] <<SYS>>
你是一个有帮助的助手。
<</SYS>>

用户的问题 [/INST] 助手的回答 </s>
```

**Llama-3 / ChatML 格式**：
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
你是一个有帮助的助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
用户的问题<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### 4.3 Apply Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "什么是机器学习？"},
]

# 应用 chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 在末尾加上 assistant 的开头
)
print(text)
```

---

## 五、Tokenizer 对 LLM 性能的影响

### 5.1 词表大小与压缩率

词表越大，同样文本需要的 token 数越少（压缩率更高）：

| 模型 | 词表大小 | 中文汉字平均 tokens/字 |
|------|--------|-------------------|
| GPT-2 | 50K | 约 1.5-2 |
| Llama-2 | 32K | 约 1.5-2 |
| Llama-3 | 128K | 约 1（一字一 token）|
| Qwen（中文优化）| 150K+ | < 1（多字合并）|

Llama-3 将词表从 32K 扩大到 128K，特别增加了更多多语言 token，显著提升了非英语语言的压缩率。

### 5.2 Tokenization 对推理的影响

```python
# 同样内容，不同 tokenizer 的 token 数差异
text = "北京是中华人民共和国的首都"

# Llama-2 tokenizer（32K 词表）
tokens_llama2 = tokenizer_llama2.encode(text)
print(len(tokens_llama2))  # 约 20 tokens

# Qwen tokenizer（150K 词表，中文优化）
tokens_qwen = tokenizer_qwen.encode(text)
print(len(tokens_qwen))  # 约 8 tokens
```

Token 数少意味着：
- 更短的序列 → 更快的推理速度
- 更小的 KV Cache → 可以在同等显存下处理更多请求
- 更低的推理成本（API 按 token 计费）

### 5.3 数字的 Tokenization 问题

BPE 对数字的处理可能不直观：

```python
# GPT-2 tokenizer
tokens = tokenizer.encode("1234567890")
# → ["12", "345", "678", "90"]（不是按位分割）
```

这会导致数学推理困难（模型难以对任意精度的数字做算术）。

**改进方案**：
- 使用 **digit-level tokenization**（每个数字一个 token）：训练成本增加，但数学能力提升
- 或者扩大词表，包含更多数字组合

---

## 六、推理框架中的 Tokenizer

### 6.1 Tokenizer 的并发挑战

在高并发推理服务中，Tokenizer 可能成为 CPU 瓶颈：

- vLLM 使用单独的 `TokenizerGroup` 异步处理 tokenization
- SGLang 有专门的 `TokenizerManager` 进程

### 6.2 Streaming 解码的 Detokenization 问题

流式输出（Streaming）时，每生成一个 token 就需要立即转换为文本。但 BPE 可能产生"不完整 token"：

```
生成 token 序列: [1234, 567, 89]
对应文本: ["你", "好", "世界"]

但如果分词是:
token 1234 → "你好"（两字一起）
→ 必须等到完整 token 才能输出

解决方案：StreamingDetokenizer 处理部分 UTF-8 字节序列，
只在字符边界处输出，避免乱码
```

---

## 七、总结

| 特性 | BPE | SentencePiece BPE | Unigram LM |
|------|-----|------------------|-----------|
| 算法方向 | 自底向上 | 自底向上 | 自顶向下 |
| 多语言支持 | 一般（需 byte-level） | 好 | 好 |
| 主要用途 | GPT 系列，Llama | Llama，PaLM | T5，ALBERT |
| OOV 处理 | Byte-level BPE 无 OOV | 无 OOV | 无 OOV |

现代 LLM tokenizer 的关键趋势：
1. **词表扩大**：从 32K 到 128K+，提升多语言压缩率
2. **Byte-level**：确保无 OOV，支持任意文本
3. **Chat Template**：标准化对话格式，方便指令微调
4. **高效实现**：Rust/C++ 实现（tiktoken、HuggingFace tokenizers）保证 CPU 不成为瓶颈

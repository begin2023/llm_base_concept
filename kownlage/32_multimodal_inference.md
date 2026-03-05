# 32. 多模态推理（Vision Encoder + LLM）

---

## 一、多模态 LLM 的基本架构

多模态 LLM（MultiModal LLM，MLLM）能够处理图像、视频、音频等非文本输入，并生成文本输出。以视觉语言模型（VLM）为例，典型架构如下：

```
图像输入
    ↓
Vision Encoder（视觉编码器）
  如：ViT-L/14, CLIP, DINOv2, SigLIP
    ↓
Visual Tokens（视觉 token）
  图像被编码为若干视觉 token
    ↓
Projector / Adapter（投影层）
  将视觉特征的维度映射到 LLM 的 hidden_dim
  如：MLP、Q-Former、Linear
    ↓
Text Tokens + Visual Tokens（拼接）
  [System Prompt tokens] + [Visual Tokens] + [User Text Tokens]
    ↓
LLM（语言模型主干）
  标准 Transformer，处理混合 token 序列
    ↓
文本输出
```

---

## 二、主要组件详解

### 2.1 Vision Encoder（视觉编码器）

**ViT（Vision Transformer）**：
- 将图像切分为固定大小的 patch（如 14×14 或 16×16 像素）
- 每个 patch 被 Flatten 后线性投影为 token embedding
- 通过 Transformer 编码，输出每个 patch 的特征

```
图像 224×224：
  patch 大小 14×14 → (224/14)² = 256 个 patch token
  每个 patch 编码为 1024 维向量（ViT-L）
  输出：256 × 1024 特征矩阵
```

**CLIP ViT（对比学习预训练）**：
- OpenAI CLIP 使用图像-文本对比学习预训练 ViT
- 学习图像和文本的对齐表示
- LLaVA、Qwen-VL 等模型使用 CLIP ViT 作为视觉编码器

**DINOv2 / SigLIP**：
- DINOv2：自监督预训练，提供更丰富的视觉特征
- SigLIP（Google）：使用 Sigmoid Loss 代替 InfoNCE 的对比学习
- 比 CLIP 在细粒度视觉理解任务上更强

### 2.2 Projector / Adapter（投影层）

**Linear Projector**：
- 最简单的设计：一个线性层
- `visual_tokens = visual_features @ W_proj`
- 维度变换：ViT 输出维度（如 1024）→ LLM hidden_dim（如 4096）

**MLP Projector（LLaVA-1.5 采用）**：
```python
# 2 层 MLP with GELU
class MLPProjector(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, visual_features):
        return self.layers(visual_features)
```

**Q-Former（BLIP-2 采用）**：
- 使用一组可学习的 Query token，通过 Cross-Attention 从 ViT 特征中提取信息
- 将 N 个 patch token 压缩为固定数量的 Query token（如 32 个）
- 大幅减少传入 LLM 的视觉 token 数量

**Resampler（Flamingo 采用）**：
- 类似 Q-Former 的设计，通过 Perceiver 将可变数量的图像特征压缩为固定数量

### 2.3 视觉 Token 的融合方式

**拼接方式（Concatenation）**：
- 将视觉 token 直接拼接到文本 token 序列中
- `[IMG_1, IMG_2, ..., IMG_256, TEXT_1, TEXT_2, ...]`
- LLM 通过正常的 Self-Attention 同时处理视觉和文本 token
- 最主流的方式（LLaVA、Qwen-VL 等）

**交叉注意力方式（Cross-Attention）**：
- 文本 token 通过 Cross-Attention 查询视觉 token
- 类似 Flamingo、Idefics 的设计
- 优点：视觉信息不占用 context window
- 缺点：需要修改 LLM 架构

---

## 三、代表性多模态模型

### 3.1 LLaVA 系列

**LLaVA-1.5**：
- Vision Encoder：CLIP ViT-L/14@336（336×336 像素输入）
- Projector：2 层 MLP
- LLM：Llama-2/Vicuna 7B/13B
- 视觉 token 数：576（24×24 个 patch）
- 特点：简单有效，开源，性能优秀

**LLaVA-NeXT（LLaVA-1.6）**：
- 引入动态高分辨率（Dynamic High Resolution）
- 将大图切分为多个 tile，每个 tile 分别编码
- 支持最高 1344×1344 分辨率
- 视觉 token 数：最多 2880

### 3.2 Qwen-VL

- Vision Encoder：自研 ViT（448×448 输入）
- Projector：Q-Former（压缩为 256 token）
- LLM：Qwen 7B
- 特点：中文多模态能力强，支持细粒度视觉理解

**Qwen2-VL**：
- 引入 Naive Dynamic Resolution：动态分辨率，视觉 token 数随图像大小变化
- 引入 2D-RoPE：专门为 2D 图像设计的 RoPE
- 支持视频输入

### 3.3 InternVL 系列

- 使用自研 InternViT（超大 ViT，6B 参数）
- 多分辨率编码策略
- 开源模型中性能最强之一

### 3.4 GPT-4V / GPT-4o

- 具体架构未公开
- 推测使用 CLIP ViT 或类似编码器
- GPT-4o 支持原生多模态（图像、音频、文本统一处理）

---

## 四、多模态推理的工程挑战

### 4.1 动态分辨率处理

不同图像的分辨率不同，会导致视觉 token 数量不同：

**处理方案**：
1. **固定分辨率**（最简单）：所有图像 Resize 到固定大小（如 336×336）
2. **动态分辨率**（LLaVA-NeXT）：根据图像原始分辨率动态决定切分 tile 数量
3. **任意分辨率**（AnyRes）：支持任意输入分辨率，通过 Q-Former 压缩

### 4.2 视频输入处理

视频 = 多帧图像序列：

```python
# 视频帧采样
def sample_frames(video_path, num_frames=8):
    frames = load_video(video_path)
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    return [frames[i] for i in indices]

# 每帧编码为视觉 token，拼接
visual_tokens = []
for frame in sampled_frames:
    frame_tokens = vision_encoder(frame)  # [256, hidden_dim]
    visual_tokens.append(frame_tokens)
visual_tokens = torch.cat(visual_tokens, dim=0)  # [256*8, hidden_dim]
```

视频理解的挑战：
- 帧数多 → 视觉 token 数量大 → KV Cache 显著增大
- 时序关系理解
- 需要时序压缩（如时序池化）

### 4.3 多图像处理

对话中可能包含多张图像，需要明确每张图像在文本序列中的位置：

```
对话格式：
"<img_1> 这张图片里有什么？ <img_2> 和这张图片有什么区别？"

处理：
token_ids = tokenize(text)  # 包含 <img_1>, <img_2> 占位符
visual_tokens_1 = encode(image_1)
visual_tokens_2 = encode(image_2)
# 将占位符替换为对应的视觉 token
```

---

## 五、多模态推理的显存估算

以 LLaVA-1.5 7B 为例（BF16）：

| 组件 | 大小 | 显存 |
|------|------|------|
| LLM（Llama-2 7B） | 7B 参数 | ~14 GB |
| Vision Encoder（CLIP ViT-L） | 0.3B 参数 | ~0.6 GB |
| MLP Projector | 微小 | < 0.1 GB |
| KV Cache（2K token） | - | ~2 GB |
| **总计** | | **~17 GB** |

对于高分辨率图像（如 LLaVA-NeXT 1344×1344）：
- 视觉 token 数 = 2880（5 个 tile × 576）
- 这些 token 都进入 LLM，显著增加 KV Cache 占用

---

## 六、推理框架的多模态支持

### 6.1 vLLM 多模态支持

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# 图像输入
image = Image.open("image.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "描述这张图片"},
        ],
    }
]

outputs = llm.chat(messages, SamplingParams(max_tokens=256))
```

### 6.2 SGLang 多模态支持

```python
import sglang as sgl

@sgl.function
def describe_image(s, image):
    s += sgl.user(sgl.image(image) + "描述这张图片")
    s += sgl.assistant(sgl.gen("description", max_tokens=256))

result = describe_image.run(image="image.jpg")
```

---

## 七、前沿发展

### 7.1 Native Multimodal（原生多模态）

GPT-4o、Gemini 等模型尝试统一处理所有模态，而不是"LLM + 外挂视觉编码器"的架构：
- 统一的 tokenizer：将图像、音频、文本都 tokenize 为统一的 token
- 端到端训练，模态间交互更自然

### 7.2 Vision Token 压缩

视觉 token 数量多是推理的主要瓶颈，近期研究方向：
- **TokenPacker**、**FastV** 等：在 LLM 推理过程中动态丢弃不重要的视觉 token
- 减少 50-80% 的视觉 token，推理速度提升 2-5×，性能损失很小

---

## 八、总结

多模态推理的核心架构：
1. **Vision Encoder**：ViT（CLIP/DINOv2/SigLIP），将图像编码为 token
2. **Projector**：MLP 或 Q-Former，桥接视觉和文本模态
3. **LLM**：标准 Transformer，统一处理多模态 token 序列

主要工程挑战：
- 动态分辨率 → 视觉 token 数不固定 → 影响 batching 效率
- 大量视觉 token → KV Cache 压力 → 需要 token 压缩技术
- 多帧视频 → 极大的 context length → 需要时序压缩

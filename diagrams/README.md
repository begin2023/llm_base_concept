# Attention 计算过程图解

用 T=4 (4个token), D=3 (head dim=3) 的例子，拆解 Attention 的两步矩阵乘法。

## 打开方式

把 `.drawio` 文件内容复制到 [app.diagrams.net](https://app.diagrams.net/) 查看：
1. 打开 https://app.diagrams.net/
2. 选择 "Extras" → "Edit Diagram"（或按 Ctrl+Shift+X）
3. 粘贴 .drawio 文件的 XML 内容
4. 点 OK

或者直接用"File → Open From → Device"打开 .drawio 文件。

## 文件说明

### 1. attention-overview.drawio — 总览

```
Q [T,D]  ──┐
            ├─ matmul ──→ S [T,T] ──→ softmax ──→ α [T,T] ──┐
K^T [D,T]──┘                                                  ├─ matmul ──→ O [T,D]
                                                   V [T,D]  ──┘

         Step 1: 2T²D FLOPs                      Step 2: 2T²D FLOPs
                            总计: 4T²D FLOPs
```

### 2. attention-step1-qkt.drawio — Step 1: S = Q × K^T

以 s₁₂ 为例展示计算过程：

```
s₁₂ = Q的第1行(q₁) · K^T的第2列(k₂) 的点积

     = q₁₀×k₂₀ + q₁₁×k₂₁ + q₁₂×k₂₂
       ~~~~~~~~   ~~~~~~~~   ~~~~~~~~
       D=3 次乘法 + 2 次加法 ≈ 2D FLOPs
```

**S[i][j] 的含义**: token i 的 query 和 token j 的 key 的相似度（原始注意力分数）

- 行 = query 的 token 位置
- 列 = key 的 token 位置
- 共 T×T = 16 个元素，每个 2D FLOPs → **Step 1 总计 2T²D**

### 3. attention-step2-attn-v.drawio — Step 2: O = α × V

以 o₁₂ 为例展示计算过程：

```
o₁₂ = α的第1行 × V的第2列

     = α₁₀×v₀₂ + α₁₁×v₁₂ + α₁₂×v₂₂ + α₁₃×v₃₂
```

**但按 (i,j) attention 点来拆分**，每个 (i,j) 对贡献的是：

```
对 D 个维度都做:  o_i[d] += α_ij × v_j[d]

   o₁[0] += α₁ⱼ × vⱼ[0]    ← 1次乘 + 1次加
   o₁[1] += α₁ⱼ × vⱼ[1]    ← 1次乘 + 1次加
   o₁[2] += α₁ⱼ × vⱼ[2]    ← 1次乘 + 1次加
                               ────────────
                               D次乘 + D次加 = 2D FLOPs

   ★ 加法来自 "+=" 累加，不是凭空出现的！
```

**O[i][d] 的含义**: 所有 token 的 value 第 d 维的加权和，权重是 token i 的注意力分布

## FLOPs 总结

```
每个 (i,j) attention 点:
   Step 1:  q_i · k_j       = D次乘 + D次加 = 2D
   Step 2:  o_i += α_ij×v_j = D次乘 + D次加 = 2D
                                            ──────
                                    合计 4D FLOPs

总量 = attention点数 × 每点FLOPs
     = (B × H × T²) × 4D
     = 4BT²HD
```

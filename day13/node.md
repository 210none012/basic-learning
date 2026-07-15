# Day 13: Minimind代码学习

## 第一部分：SwiGLU（门控激活前馈网络）

### 一、传统 FFN vs SwiGLU

#### 传统 FFN
```
x → Linear → SiLU → Linear → output
```

#### SwiGLU（门控线性单元 + Swish 激活）
```
        ┌→ Linear(gate) → SiLU ─┐
x ──────┤                        × → Linear(down) → output
        └→ Linear(up) ─────────┘
```

$$
\text{SwiGLU}(x) = (\text{SiLU}(xW_g)) \odot (xW_u) \cdot W_d
$$

### 二、为什么 SwiGLU 更好

| 优势 | 说明 |
|------|------|
| **门控机制** | `gate_proj` 决定哪些信息通过，类似 LSTM 的门控思想 |
| **非线性更强** | `SiLU(gate) × up` 比单纯 `SiLU(up)` 表达能力更强 |
| **梯度流动** | 门控有助于缓解梯度消失，信息选择性通过 |
| **参数量** | 三个权重矩阵（gate/up/down），参数量约为传统 FFN 的 1.5 倍 |

### 三、SiLU（Swish）激活函数

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

| 特性 | 说明 |
|------|------|
| 平滑非单调 | 负区间有微小负值，保留部分负信息 |
| 无界上界 | 正区间不会饱和 |
| 有界下界 | 负区间趋近于 -0.278 |

---

## 第二部分：MoE（混合专家）

### 一、核心思想

将一个线性层替换为**多个专家**（多个并行线性层），每个 token 只激活其中少数专家：

```
x → Router → 选 Top-K 专家 → 各专家独立计算 → 加权求和 → 输出
```

### 二、MiniMind 的 MoE 配置

| 参数 | 默认值 | 说明 |
|------|------|------|
| `num_experts` | 4 | 专家总数 |
| `num_experts_per_tok` | 1 | 每个 token 激活几个专家 |
| `router_aux_loss_coef` | 5e−4 | 负载均衡损失的权重 |

### 三、Router（路由）

```python
# 1. 计算每个 token 对每个专家的匹配分数
router_logits = gate(x)  # (batch, seq, num_experts)

# 2. Top-K 选择
top_k_logits, top_k_indices = router_logits.topk(k=num_experts_per_tok)

# 3. Softmax 归一化选中的专家权重
weights = softmax(top_k_logits)

# 4. 各专家计算后加权求和
output = sum(weight_i * expert_i(x) for i in selected)
```

### 四、保持未选中专家的可训练性

**问题**：当某专家没有被任何 token 选中时，其参数不参与前向计算，也不在计算图中，无法更新。

**解决方案**：强制未被选中的专家也参与一次计算，前向结果加 0：

```python
for expert in experts:
    expert_output = expert(x)
    output += gate_mask * expert_output
# gate_mask = 0 时，expert_output 参与计算图但不影响结果
```

> 这样未被选中的专家参数仍在计算图中，反向传播时梯度正常流动。

### 五、负载均衡损失（Auxiliary Loss）

**问题**：Router 可能退化——所有 token 都选同一个专家，其余专家形同虚设。

**损失公式**：

$$aux\_loss = N \cdot \alpha \cdot \sum_{i=1}^{N} (l_i \cdot s_i)$$

| 符号 | 含义 |
|------|------|
| $N$ | 专家数量 |
| $\alpha$ | 损失系数（`router_aux_loss_coef`） |
| $l_i$ | 专家 i 被选中的平均概率（load） |
| $s_i$ | 专家 i 的路由分数均值（score） |

**直觉**：
- 某专家分数过高 → $l_i \cdot s_i$ 大 → 损失大 → 梯度惩罚
- 各专家均衡 → $l_i \cdot s_i$ 差异小 → 损失小

---

## 第三部分：Attention 前向传播详解

### 一、完整 forward 流程

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape

    # 1. QKV 投影
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    # 2. 重塑为多头形状
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

    # 3. QK RMSNorm + RoPE
    xq, xk = self.q_norm(xq), self.k_norm(xk)
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

    # 4. 拼接 KV Cache（推理时逐个生成 token）
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)

    # 5. GQA：扩展 KV 头 + 转置为 (batch, heads, seq, head_dim)
    xq, xk, xv = (xq.transpose(1, 2),
                   repeat_kv(xk, self.n_rep).transpose(1, 2),
                   repeat_kv(xv, self.n_rep).transpose(1, 2))

    # 6. Flash Attention 或标准注意力
    if self.flash and seq_len > 1 and ... :
        output = F.scaled_dot_product_attention(xq, xk, xv, ...)
    else:
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果掩码
        if self.is_causal:
            scores[:,:,:, -seq_len:] += torch.full((seq_len, seq_len), float("-inf")).triu(1)
        # Padding 掩码
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

    # 7. 合并多头 → 输出投影
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
    output = self.resid_dropout(self.o_proj(output))
    return output, past_kv
```

### 二、关键细节解析

| 步骤 | 操作 | 说明 |
|------|------|------|
| QKV 投影 | `nn.Linear` | 各自独立投影，KV 头数 ≤ Q 头数 |
| `view` 重塑 | → (batch, seq, heads, head_dim) | 分离多头维度 |
| QK Norm | RMSNorm | Q、K 各自做归一化 |
| RoPE | `apply_rotary_pos_emb` | 注入位置信息 |
| KV Cache | `torch.cat` 拼接历史 | 推理时分步生成，避免重复计算 |
| `repeat_kv` | KV 头数扩展至 Q 头数 | GQA 的核心操作 |
| `transpose(1,2)` | → (batch, heads, seq, head_dim) | 适配注意力计算格式 |

### 三、Flash Attention vs 标准注意力

| | Flash Attention | 标准注意力 |
|------|------|------|
| 条件 | PyTorch ≥ 2.0 且 seq_len > 1 | 兜底方案 |
| S = QK^T | 分块计算，不存储完整矩阵 | 显存中存储完整矩阵 |
| 精度 | float（自动） | `scores.float()` 手动提升 |
| 速度 | **快数倍** | 基线 |

### 四、两种掩码

#### 因果掩码（Causal Mask）

```python
scores[:,:,:, -seq_len:] += torch.full((seq_len, seq_len), float("-inf")).triu(1)
```

- `triu(1)`：生成上三角矩阵，保留对角线以上元素
- 效果：第 i 个 token 只能看到前 i 个 token（包括自己）
- 只在最后 `seq_len` 列添加（配合 KV Cache，前面是历史缓存）

#### Padding 掩码

```python
scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
```

- `attention_mask`：1 表示有效 token，0 表示 padding
- `(1-mask) * -1e9`：padding 位置的 scores 变为极大负数
- softmax 后 padding 位置权重趋近于 0

---

## 第四部分：FeedForward（SwiGLU 实现）

```python
class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size=None):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 数据流

```
x → gate_proj → SiLU ──┐
                         × → down_proj → output
x → up_proj ────────────┘
```

---

## 第五部分：生成（Generate）流程

### 完整自回归生成

```python
@torch.inference_mode()
def generate(self, inputs, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, ...):
    for _ in range(max_new_tokens):
        # 1. 只传入新增 token（配合 KV Cache）
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(input_ids[:, past_len:], ...)

        # 2. 取最后一个 token 的 logits，除以温度
        logits = outputs.logits[:, -1, :] / temperature

        # 3. 重复惩罚
        if repetition_penalty != 1.0:
            for i in range(batch):
                seen = torch.unique(input_ids[i])
                logits[i, seen] = torch.where(
                    score > 0, score / repetition_penalty,
                    score * repetition_penalty
                )

        # 4. Top-K + Top-P 采样
        if top_k > 0:
            logits[logits < top_k_threshold] = -float('inf')
        if top_p < 1.0:
            # 按概率累积，超过 p 的 token 置 -inf

        # 5. 采样或贪心
        next_token = torch.multinomial(softmax(logits)) if do_sample else argmax(logits)

        # 6. 拼接新 token
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 7. 检查是否全部遇到 EOS
        if finished.all(): break
    return input_ids
```

### 采样参数

| 参数 | 默认 | 作用 |
|------|------|------|
| `temperature` | 0.85 | 控制随机性（>1 更随机，<1 更确定） |
| `top_k` | 50 | 只保留概率最高的 k 个 token |
| `top_p` | 0.85 | 保留累积概率达 p 的最小 token 集 |
| `repetition_penalty` | 1.0 | >1 惩罚重复，<1 鼓励重复 |
| `do_sample` | True | True=多项式采样，False=贪心解码 |
| `max_new_tokens` | 8192 | 最大生成长度 |

### 训练时的 Loss 计算

```python
# 用前面的已知量预测下一个 token
# logits 去掉最后一个 → labels 去掉第一个
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

> `ignore_index=-100`：忽略标签为 -100 的位置（padding 或特殊标记）。

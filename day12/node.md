# Day 12: MiniMind 源码解析 
## 第一部分：LoRA（Low-Rank Adaptation）

### 一、核心原理

用两个低秩矩阵乘积替代全参数更新：

```
h = Wx + ΔWx = Wx + BAx
```

- **A**：降维矩阵 (in×rank)，高斯初始化（保证初始权重合理）
- **B**：升维矩阵 (rank×out)，全零初始化（保证旁路初始输出为0）

> B 全零初始化确保从原模型开始微调，不破坏已有能力。

### 二、LoRA 网络结构（model_lora.py）

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维
        self.A.weight.data.normal_(mean=0, std=0.02)         # 高斯初始化
        self.B.weight.data.zero_()                            # 全零初始化

    def forward(self, x):
        return self.B(self.A(x))
```

### 三、四个关键操作

#### 1. apply_lora — 挂载 LoRA

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and module.in_features == module.out_features:
        lora = LoRA(module.in_features, module.out_features, rank=16)
        setattr(module, "lora", lora)
        # 替换 forward：原始输出 + LoRA 旁路输出
```

> 只对方阵（输入维度=输出维度）的线性层添加 LoRA，用于 Attention 的 QKV 投影。

#### 2. save_lora — 只保存 LoRA 权重

```python
# 体积远小于全量模型（仅 rank × (in+out) 参数）
raw_model = getattr(model, '_orig_mod', model)
for name, module in raw_model.named_modules():
    if hasattr(module, 'lora'):
        lora_state = {f'{name}.lora.{k}': v.cpu().half() for ...}
torch.save(state_dict, path)
```

#### 3. load_lora — 加载 LoRA 权重

```python
model.load_lora(path)
# 自动处理 nn.DataParallel 的 module. 前缀
```

#### 4. merge_lora — 合并 LoRA 到原权重

```python
# 加载 LoRA → 将 BA 加到 W 上 → 保存为独立模型
state_dict[f'{name}.weight'] += (module.lora.B.weight @ module.lora.A.weight)
```

> `getattr(model, '_orig_mod', model)` — 有 `_orig_mod` 则返回，否则返回原始模型。`torch.compile()` / `torch.amp()` 等方法可能添加此属性。

---

## 第二部分：MiniMind 模型核心组件

### 一、Config 配置（MiniMindConfig）

| 参数 | 默认值 | 说明 |
|------|------|------|
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `vocab_size` | 6400 | 词表大小 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA） |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `rope_theta` | 1e6 | RoPE 基数 |
| `use_moe` | False | 是否启用 MoE |
| `flash_attn` | True | Flash Attention 加速 |
| `tie_word_embeddings` | True | 输入/输出词嵌入共享 |

#### MoE 相关

| 参数 | 默认 | 说明 |
|------|------|------|
| `num_experts` | 4 | 专家数量 |
| `num_experts_per_tok` | 1 | 每个 token 选几个专家 |
| `router_aux_loss_coef` | 5e−4 | 路由辅助损失（平衡专家负载） |

### 二、RMSNorm

| | LayerNorm | RMSNorm |
|------|------|------|
| 操作 | 减均值 + 除标准差 | 只除 RMS |
| 平移参数 | 有（可学习） | 无 |
| 缩放参数 | 有 | 有 |

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma$$

```python
class RMSNorm(nn.Module):
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

> 先转 `float` 保证数值稳定性，再转回原类型。

### 三、GQA（分组查询注意力）

| | MHA | GQA |
|------|------|------|
| KV 头 | = Q 头数 | < Q 头数 |
| 关系 | 一一对应 | 多个 Q 头共享一组 KV |
| 效果 | — | 大幅减少 KV Cache |

```python
# Q头=8, KV头=4 → n_rep=2（每2个Q头共享1组KV）
def repeat_kv(x, n_rep):
    # (bs, seq, num_kv, head_dim) → (bs, seq, num_kv*n_rep, head_dim)
    return x[:,:,:,None,:].expand(bs, seq, n_kv, n_rep, head_dim)\
           .reshape(bs, seq, n_kv*n_rep, head_dim)
```

---

## 第三部分：RoPE 实现与 YaRN 扩展

### 一、RoPE 预计算

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
# 即 θ_i = 1 / 1000000^(2i/d)
frets_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
frets_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

### 二、应用 RoPE

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # 后半部分取反与前半部分拼接（旋转矩阵乘法等价形式）
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

> RoPE 隔一个数为一组 (0,2)、(1,3) 进行计算。

### 三、YaRN — 长上下文外推

#### 问题

实际推理长度超过训练长度时，RoPE 的旋转角度超出模型见过的范围。

#### 演进路线

| 方法 | 做法 | 问题 |
|------|------|------|
| **线性缩放（PI）** | 超长位置按比例压缩 | 压缩邻近距离，干扰局部分辨率 |
| **NTK-aware** | 低频内插、高频外推 | 外推引入了未见过的旋转角度 |
| **NTK-by-parts** | 按波长阈值决定内插/外推 | 阈值边界不连续 |
| **YaRN** | NTK-by-parts + 注意力缩放 | — |

#### YaRN 核心公式

- 设定两个阈值：`beta_fast`（高频阈值）和 `beta_slow`（低频阈值）
- 小于阈值：完全内插（绝对位置信息多）
- 大于阈值：完全外推（相对位置信息多）
- 中间：线性混合

- 注意力缩放修正：`Attention = softmax(qK^T / (t√|D|))`，`√1/t = 0.1·ln(s) + 1`

#### YaRN 配置

```python
rope_scaling = {
    "beta_fast": 32,    # 高频阈值，大于此值不缩放
    "beta_slow": 1,      # 低频阈值，小于此值不缩放
    "factor": 16,        # 扩张倍数
    "original_max_position_embeddings": 2048,  # 训练时的最大长度
    "type": "yarn"
}
```

---

## 四、Attention 层配置

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)     # Q/K 加 RMSNorm
        self.k_norm = RMSNorm(head_dim)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn
```

---

## 五、要点总结

1. **LoRA**：BA 分解，A 高斯初始化 + B 零初始化，从原模型开始微调
2. **RMSNorm**：比 LayerNorm 少平移参数，计算更高效
3. **GQA**：减少 KV 头，多个 Q 头共享 KV，大幅压缩 KV Cache
4. **RoPE**：旋转矩阵编码相对位置，隔维分组计算
5. **YaRN**：NTK-by-parts + 注意力缩放，实现训练长度的 16 倍外推

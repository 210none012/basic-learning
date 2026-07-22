# Day 18: MiniMind代码阅读 与 Transformer-XL

## 第一部分：混合精度训练

### 一、为什么需要混合精度

| 对比维度  | 纯 FP32 | 纯 FP16   | **混合精度（AMP）**   |
| ----- | ------ | -------- | --------------- |
| 速度    | 基准(1x) | 极快(2x+)  | 很快(1.5x~2x)     |
| 收敛性   | 稳定     | 极差（梯度下溢） | 几乎稳定            |
| 显存    | 高      | 低        | 低（减半）           |
| 代码复杂度 | 简单     | 复杂（需手调）  | 极简（PyTorch 已封装） |

### 二、GradScaler 工作流程

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

with autocast_ctx:
    loss = model(input_ids, labels=labels).loss

# 1. 损失缩放后反向传播
scaler.scale(loss).backward()

# 2. 反缩放得到正确梯度
scaler.unscale_(optimizer)

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(params, max_norm)

# 4. 更新参数
scaler.step(optimizer)
scaler.update()     # 动态调整缩放因子

# 5. 清零梯度
optimizer.zero_grad(set_to_none=True)
```

### 三、关键细节

| 步骤 | 作用 |
|------|------|
| `scaler.scale(loss)` | 将小 loss 放大，防止 FP16 下溢 |
| `unscale_()` | 反缩放，恢复真实梯度用于裁剪 |
| `scaler.step()` | 用正确梯度的 FP32 副本更新参数 |
| `scaler.update()` | 根据溢出情况动态调整缩放因子 |
| `bfloat16` vs `float16` | bf16 动态范围更大，无需 GradScaler |

---

## 第二部分：Log 概率计算（rollout_engine.py）

### 为什么用 Log 概率

|      | 原始概率         | Log 概率         |
| ---- | ------------ | -------------- |
| 数值范围 | (0, 1)，连乘→下溢 | (-∞, 0]，加法替代乘法 |
| 稳定性  | 差            | 好              |
| 梯度   | 小概率→小梯度      | 稳定合理           |

### 实现

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    # 1. 前向得到 logits，去掉最后一位
    logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    # 2. 对每个 token，取它对应 id 的 log 概率
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

**关键**：`logits[:-1]` 对应输入 `ids[1:]`（第 i 个 logit 预测第 i+1 个 token）。

---

## 第三部分：LoRA 训练流水线（train_lora.py）

### 训练流程

```python
for epoch in range(epochs):
    for step, (input_ids, labels) in enumerate(loader):
        # 1. 学习率调整
        lr = get_lr(epoch * iters + step, epochs * iters, learning_rate)

        # 2. 前向 + 混合精度
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = (res.loss + res.aux_loss) / accumulation_steps

        # 3. 缩放 → 反向
        scaler.scale(loss).backward()

        # 4. 梯度累积步数到后更新参数
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(lora_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 5. 日志 + 保存
        Logger(f'loss: {loss:.4f}, lr: {lr:.8f}')
        save_lora(model, save_path)
```

### 关键设计

| 设计 | 说明 |
|------|------|
| **梯度累积** | `loss / accumulation_steps` → 小 batch 模拟大 batch |
| **只训 LoRA** | 冻结非 LoRA 参数，`requires_grad` 区分 |
| **断点续训** | 从 checkpoint 恢复 model/optimizer/scaler 状态 |
| **分布式** | `DistributedDataParallel` + `DistributedSampler` |
| **LoRA 保存** | 只保存 LoRA 权重，极小体积 |


---

## 第四部分：Rollout 引擎（rollout_engine.py）

### 设计模式

```python
# 抽象基类
class RolloutEngine(ABC):
    @abstractmethod
    def rollout(self, prompt_ids, ...) -> RolloutResult: pass
    @abstractmethod
    def update_policy(self, model): pass

# PyTorch 实现
class TorchRolloutEngine(RolloutEngine):
    def rollout(self, ...):
        with torch.no_grad(), autocast_ctx:
            output_ids = model.generate(prompt_ids, ...)
            completion_ids = output_ids[:, prompt_len:]
            per_token_logps = compute_per_token_logps(...)
        # 返回生成结果 + 每个 token 的 log 概率
        return RolloutResult(output_ids, completion_ids, per_token_logps, ...)
```

### RolloutResult

| 字段 | 说明 |
|------|------|
| `output_ids` | 完整序列 (prompt + completion) |
| `completion_ids` | 仅生成部分 |
| `per_token_logps` | 每个生成 token 的 log 概率 |
| `completions` | 解码后的文本 |

---

## 第五部分：Transformer-XL

### 一、核心问题

标准 Transformer 在固定长度段上独立训练，段之间无信息流动：
- 无法捕捉超出预设长度的**长期依赖**
- **上下文碎片化**：强行切分不考虑语义边界

### 二、核心方案：段级循环

| 组件 | 标准 Transformer | Transformer-XL |
|------|------|------|
| Query | 当前片段 | 当前片段 |
| Key | 当前片段 | 当前 + 上一片段**记忆** |
| Value | 当前片段 | 当前 + 上一片段**记忆** |
| 梯度 | 完整传播 | **不传到上一片段**（stop-gradient） |

$$
\tilde{\mathbf{h}}_{\tau+1}^{n-1} = [\text{SG}(\mathbf{h}_{\tau}^{n-1}) \circ \mathbf{h}_{\tau+1}^{n-1}]
$$

### 三、依赖长度

| 标准 | XL |
|------|------|
| = 片段长度 L | = **N × L**（N 为层数） |

### 四、相对位置编码

标准注意力分四项（内容+位置交叉），X元-L 重新参数化为相对位置：

$$\mathbf{A}_{i,j}^{\text{rel}} = E_{x_i}^T W_q^T W_{k,E} E_{x_j} + E_{x_i}^T W_q^T W_{k,R} R_{i-j} + u^T W_{k,E} E_{x_j} + v^T W_{k,R} R_{i-j}$$

| 符号 | 含义 |
|------|------|
| $R_{i-j}$ | 正弦相对位置编码（固定，无可学习参数） |
| $W_{k,E}, W_{k,R}$ | 分离内容 Key 和位置 Key |
| $u, v$ | 可训练偏置（替代绝对 Query） |

> 实际通过矩阵移位降至线性复杂度。

### 五、对比总结

| 维度 | 标准 Transformer | Transformer-XL |
|------|------|------|
| 段间关系 | 独立，无信息流 | 复用上段隐藏状态 |
| 最大依赖 | = 段长 | = 层数 × 段长 |
| 上下文碎片 | 严重 | 解决 |
| 评估速度 | 极慢（每次重算） | 极快（缓存复用） |
| 位置编码 | 绝对 | **相对** |

---

## 六、要点总结

1. **混合精度**：AMP + GradScaler 接近 FP16 速度但保持 FP32 稳定性
2. **Log 概率**：`log_softmax + gather`，加法替代乘法，避免数值下溢
3. **LoRA 训练**：梯度累积、断点续训、只保存 LoRA 权重
4. **Rollout 引擎**：抽象基类 + PyTorch 实现，输出每 token log 概率
5. **Transformer-XL**：段级循环 + 相对位置编码，依赖长度 O(NL)

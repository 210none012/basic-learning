# Day 20: DeepSeek-V2 技术报告精读

> 论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

---

## 第一部分：总体架构

DeepSeek-V2 采用 Transformer 架构，每个 Transformer Block 包含一个注意力模块和一个 FFN 模块。两大核心创新：

- **MLA（Multi-head Latent Attention）**：大幅压缩 KV Cache，降低推理显存
- **DeepSeekMoE**：细粒度专家分割 + 共享专家隔离，高效利用参数

![架构图](https://i.postimg.cc/6qvRyTMZ/image.png)

### 关键参数

| 参数 | 取值 |
|------|:---:|
| $N_s$（共享专家数） | 2 |
| $N_r$（路由专家数） | 160 |
| $K_r$（每 Token 激活数） | 6 |
| 隐藏维度 $d$ | 5120 |
| 专家中间隐层维度 | 1536 |
| MoE 层位置 | 除第一层外所有 FFN |

---

## 第二部分：MLA —— 多头潜在注意力

![MLA 结构图](https://i.postimg.cc/s2T00sRg/image.png)

### 一、背景：KV Cache 是推理瓶颈

传统多头注意力（MHA）在推理时需要缓存每个 Token 的 Key 和 Value，KV Cache 大小为：

$$\text{KV Cache} = 2 \cdot n_h \cdot d_h \cdot L \quad \text{（L 为层数）}$$

随着序列长度增加，KV Cache 成为显存瓶颈。

| 改进方案 | 做法 | 问题 |
|----------|------|------|
| GQA（Grouped Query Attention） | 一组 Query 共享一个 KV 对 | 减少缓存但可能降低性能 |
| MQA（Multi-Query Attention） | 所有 Query 共享一个 KV 对 | 缓存最小但性能下降明显 |
| **MLA** | 低秩压缩 KV 到潜在空间 | 大幅压缩缓存，性能无损 |

### 二、MLA 核心：低秩联合压缩

MLA 的核心思想是**不直接缓存高维的 K 和 V**，而是引入一个**低维压缩潜在向量** $\mathbf{c}_t^{KV}$，推理时仅缓存这个压缩向量，使用时再"解压"重构。

**压缩阶段**（推理时只需缓存此向量）：

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

- $W^{DKV} \in \mathbb{R}^{d_c \times d}$：降维投影矩阵
- $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$：压缩后的低维潜在向量
- $d_c \ll d_h \cdot n_h$：压缩维度远小于原始 KV 维度

**重构阶段**（前向计算时展开）：

$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}$$

$$\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$

- $W^{UK} \in \mathbb{R}^{d_h n_h \times d_c}$：升维重构 Key
- $W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$：升维重构 Value

| 对比维度 | 标准 MHA | MLA |
|---------|----------|-----|
| KV Cache 大小 | $d_h \cdot n_h$ | $d_c$ |
| 压缩比 | 1× | $d_h n_h / d_c$ 倍 |

### 三、推理加速：矩阵吸收（关键工程技巧）

推理时无需显式计算 K 和 V，通过**矩阵乘法结合律**将投影矩阵融合：

$$
\text{注意力分数} = Q \cdot K^T = (X W^Q) \cdot (\mathbf{c}^{KV} \cdot W^{UK})^T = X \cdot \underbrace{(W^Q \cdot W^{UK,T})}_{\text{融合矩阵}} \cdot \mathbf{c}^{KV,T}
$$

| 吸收方式 | 效果 |
|---------|------|
| $W^Q$ 吸收 $W^{UK}$ | 直接由压缩向量 $\mathbf{c}^{KV}$ 计算注意力分数 |
| $W^O$ 吸收 $W^{UV}$ | 注意力输出直接还原，无需显式构造 V |

> 既**节省存储**（KV Cache 大幅缩减），又**节省算力**（省去 K/V 的显式计算）。

### 四、解耦 RoPE（位置编码兼容）

低秩压缩与 RoPE 存在兼容性问题（RoPE 对 K 做旋转变换会破坏压缩结构），MLA 采用**解耦策略**：

- $\mathbf{c}_t^{KV}$ 重构出的 $\mathbf{k}_t^C$ 和 $\mathbf{v}_t^C$ **不携带位置信息**
- 额外引入**小维度**的 $\mathbf{q}_t^R$ 和共享的 $\mathbf{k}_t^R$，专门承载 RoPE
- 每个头的 Query 和 Key 为拼接形式：

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]$$

推理时实际缓存内容：

$$\text{总 KV Cache 元素数} = (d_c + d_h^R) \cdot l$$

其中 $d_h^R$ 是 RoPE 部分的每头维度（通常很小），$l$ 为层数。

### 五、MLA 优势总结

| 维度       | 传统 MHA             | MLA             |
| -------- | ------------------ | --------------- |
| KV Cache | $d_h \cdot n_h$ 每头 | $d_c$（压缩后）      |
| 显存占用     | 高                  | 大幅降低            |
| 计算量      | K/V 显式构造           | 矩阵吸收，省算力        |
| 位置编码     | RoPE 直接作用于 K       | 解耦 RoPE，压缩与位置分离 |

---

## 第三部分：DeepSeekMoE 架构

### 一、两个关键设计理念

DeepSeekMoE 在传统 MoE 基础上提出两项创新：

| 设计理念 | 具体做法 | 目的 |
|---------|---------|------|
| **细粒度专家分割**（Fine-grained Expert Segmentation） | 将专家切分为更小粒度（160 个小专家 vs 传统 8-16 个） | 提高专家专业化程度，实现更精准的知识获取 |
| **共享专家隔离**（Shared Expert Isolation） | 隔离出 2 个共享专家，始终激活 | 缓解路由专家之间的知识冗余，捕获通用知识 |

> 相同激活参数量和总参数量下，DeepSeekMoE 性能显著优于 GShard 等传统 MoE 架构。

### 二、FFN 前向计算公式（完整形式）

$$
\boxed{
\begin{aligned}
s_{i,t} &= \frac{\exp(\mathbf{u}_t^T \mathbf{e}_i)}{\sum_{j=1}^{N_r} \exp(\mathbf{u}_t^T \mathbf{e}_j)} \\[6pt]
\mathcal{I}_t &= \text{Topk}(\{s_{1,t}, \dots, s_{N_r,t}\}, K_r) \\[6pt]
g_{i,t} &= 
\begin{cases}
s_{i,t}, & i \in \mathcal{I}_t \\
0, & i \notin \mathcal{I}_t
\end{cases} \\[6pt]
\mathbf{h}_t' &= \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t)
\end{aligned}
}
$$

### 三、逐步计算流程

#### Step 1：计算亲和度

对每个路由专家 $i \in \{1, 2, \dots, N_r\}$，用 Token 与专家中心向量的点积计算亲和度：

$$s_{i,t} = \frac{\exp(\mathbf{u}_t^T \mathbf{e}_i)}{\sum_{j=1}^{N_r} \exp(\mathbf{u}_t^T \mathbf{e}_j)}$$

其中 $\mathbf{e}_i \in \mathbb{R}^d$ 为第 $i$ 个专家的**可学习中心向量**。

#### Step 2：Top-K 门控选择

从 $N_r = 160$ 个路由专家中选出亲和度最高的 $K_r = 6$ 个：

$$\mathcal{I}_t = \text{Topk}\left(\{s_{1,t}, s_{2,t}, \dots, s_{N_r,t}\}, K_r\right)$$

门控值即亲和度分数本身（不是二值化 0/1），选中专家按权重加权：

$$g_{i,t} = \begin{cases} s_{i,t}, & i \in \mathcal{I}_t \\ 0, & i \notin \mathcal{I}_t \end{cases}$$

#### Step 3：共享专家输出

2 个共享专家**始终激活**，对每个 Token 参与计算：

$$\mathbf{o}^{(s)} = \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t)$$

#### Step 4：路由专家输出

仅被选中的 6 个路由专家参与计算，按门控值加权：

$$\mathbf{o}^{(r)} = \sum_{i \in \mathcal{I}_t} s_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t)$$

#### Step 5：残差连接

$$\mathbf{h}_t' = \mathbf{u}_t + \mathbf{o}^{(s)} + \mathbf{o}^{(r)}$$

### 四、完整符号速查表

| 符号 | 类型 | 含义 |
|------|------|------|
| $\mathbf{u}_t$ | $\mathbb{R}^d$ | 第 $t$ 个 Token 的 FFN 输入 |
| $\mathbf{h}_t'$ | $\mathbb{R}^d$ | 第 $t$ 个 Token 的 FFN 输出 |
| $\mathbf{e}_i$ | $\mathbb{R}^d$ | 第 $i$ 个路由专家的中心向量（可学习） |
| $s_{i,t}$ | $\mathbb{R}$ | Token-专家亲和度分数（Softmax 归一化） |
| $g_{i,t}$ | $\mathbb{R}$ | 门控值（选中 = $s_{i,t}$，否则 = 0） |
| $\mathcal{I}_t$ | 集合 | 选中的 Top-$K_r$ 专家索引集合 |
| $N_s$ | 超参数 | 共享专家数（= 2） |
| $N_r$ | 超参数 | 路由专家数（= 160） |
| $K_r$ | 超参数 | 每 Token 激活的路由专家数（= 6） |
| $\text{FFN}_i^{(s)}$ | 函数 | 第 $i$ 个共享专家（始终激活） |
| $\text{FFN}_i^{(r)}$ | 函数 | 第 $i$ 个路由专家（按需激活） |

### 五、设备受限路由（Device-Limited Routing）

**问题**：专家并行时，160 个专家分布到多个 GPU。由于专家粒度过细，Top-6 可能需要激活分布在 6 个不同设备上的专家，导致跨设备通信成本激增。

**方案**：限制在最多 $M$ 个设备上选择亲和度最高的 $K_r$ 个专家。

**实验结论**：$M \geq 3$ 即可达到与无限制相同的效果，有效降低通信开销。

---

## 第四部分：负载均衡辅助损失

路由策略需要额外考虑**负载均衡**，防止路由坍塌（少数专家被过度使用）。DeepSeek-V2 设计了三种辅助损失：

### 一、专家级负载均衡损失（Expert-Level Balance Loss）

$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i$$

$$f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i)$$

$$P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}$$

| 符号 | 含义 |
|------|------|
| $f_i$ | 第 $i$ 个专家的平均负载（归一化后理想值 = 1） |
| $P_i$ | 第 $i$ 个专家的平均路由概率 |
| $\mathbb{1}$ | 指示函数：Token 选中该专家 = 1，否则 = 0 |

> 当 $f_i$ 和 $P_i$ 都均衡时，$\sum f_i P_i$ 最小。

### 二、设备级负载均衡损失（Device-Level Balance Loss）

$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'$$

$$f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \quad P_i' = \sum_{j \in \mathcal{E}_i} P_j$$

| 与专家级对比 | Expert-Level | Device-Level |
|------------|--------------|--------------|
| 粒度 | 单个专家 | 单个设备（一组专家） |
| 解决的问题 | 路由坍塌 | 设备间计算负载不均 |
| $P$ 的计算 | 单个专家路由概率 | 设备上所有专家 $P$ 之和 |

### 三、通信均衡损失（Communication Balance Loss）

$$\mathcal{L}_{\text{CommBal}} = \alpha_3 \sum_{i=1}^{D} f_i'' P_i''$$

- $f_i''$：第 $i$ 个设备接收 Token 的频率
- $P_i'' = P_i'$：该设备的总路由概率

**为什么需要？** Device-Limited Routing 只保证**发送**有上限，不保证**接收**均衡。该损失鼓励每个设备接收约 $M \times T$ 个 Token，确保双向通信均衡。

### 四、三种损失对比

| 损失 | 控制目标 | 粒度 |
|------|----------|------|
| $\mathcal{L}_{\text{ExpBal}}$ | 专家被均匀使用 | 单个专家 |
| $\mathcal{L}_{\text{DevBal}}$ | 各设备计算量均衡 | 设备 |
| $\mathcal{L}_{\text{CommBal}}$ | 发送/接收通信均衡 | 设备 |

总损失 = 主损失 + $\mathcal{L}_{\text{ExpBal}} + \mathcal{L}_{\text{DevBal}} + \mathcal{L}_{\text{CommBal}}$

---

## 第五部分：两阶段 RL 训练策略

### 一、为什么分两阶段？

| 数据类型 | 训练特性 |
|----------|----------|
| 推理数据（代码、数学） | 能力可在较长训练步数中**持续提升** |
| 通用数据（对话、写作） | 训练特性与推理不同，需单独处理 |

### 二、GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek-V2 使用的 RL 优化算法。与 PPO 的关键区别：

| 对比维度 | PPO | GRPO |
|----------|-----|------|
| Critic 模型 | 需要，与策略模型同规模 | **不需要** |
| 基线估计 | 依赖 Critic 模型输出 | 组内分数的均值和标准差 |
| 显存占用 | 2 倍模型大小 | 1 倍模型大小 |
| 训练成本 | 高 | **显著降低** |

**优势值计算**（组内奖励归一化）：

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}$$

通过组内**相对**表现替代 Critic 模型的绝对评分。

### 三、阶段一：推理对齐（Reasoning Alignment）

```text
Code/Math 问题 q
    │
    ▼
Policy Model 采样 G 个输出 {o1, o2, ..., oG}
    │
    ▼
RM_reasoning 打分：r_i = RM_reasoning(o_i)
    │
    ▼
GRPO 优化（利用组内相对奖励）
    │
    ▼
代码和数学能力持续提升
```

### 四、阶段二：人类偏好对齐（Human Preference Alignment）

采用**多奖励框架**，从三个维度综合评估：

$$r_i = c_1 \cdot RM_{\text{helpful}}(o_i) + c_2 \cdot RM_{\text{safety}}(o_i) + c_3 \cdot RM_{\text{rule}}(o_i)$$

| 奖励模型 | 功能 | 评估维度 |
|----------|------|----------|
| $RM_{\text{helpful}}$ | 评估**帮助性** | 是否完整回答？信息是否准确？逻辑是否清晰？ |
| $RM_{\text{safety}}$ | 评估**安全性** | 是否包含有害内容？偏见歧视？违反伦理？ |
| $RM_{\text{rule}}$ | 评估**规则遵循** | 格式是否正确？是否满足指令约束？ |

三个权重系数 $c_1, c_2, c_3$ 需要在实验中调优，平衡三维度相对重要性。

### 五、多奖励框架工作流

```text
输入问题 q
    │
    ▼
Policy Model 采样 G 个输出 {o1, ..., oG}
    │
    ├── RM_helpful(o_i)     → 帮助性评分
    ├── RM_safety(o_i)      → 安全性评分
    └── RM_rule(o_i)        → 规则遵循评分
    │
    ▼
加权组合：r_i = c1·RM_helpful + c2·RM_safety + c3·RM_rule
    │
    ▼
GRPO 优化：A_i = (r_i - mean) / std
    │
    ▼
模型参数更新 → 提升帮助性 + 安全性
```

---

## 第六部分：今日学习总结

| 模块 | 核心收获 | 与已学知识的联系 |
|------|---------|-----------------|
| MLA | 低秩压缩 KV Cache → 矩阵吸收加速 → 解耦 RoPE | day4 RoPE、day12 GQA |
| DeepSeekMoE | 160 专家 + 2 共享 → Top-6 激活 → 门控路由 | day13 MoE + SwiGLU |
| 设备受限路由 | 限制最多 M 个设备选专家，M≥3 即可 | 分布式训练通信优化 |
| 三种辅助损失 | 专家级 → 设备级 → 通信级，三层均衡 | day13 辅助损失 $aux\_loss$ |
| GRPO | 组内相对奖励替代 Critic，节省一半显存 | RLHF / PPO 基础 |
| 两阶段 RL | 先推理对齐（代码+数学），再偏好对齐（多奖励） | 训练三阶段（day6） |
| 多奖励框架 | 帮助性 + 安全性 + 规则遵循三维评估 | RLHF reward model |


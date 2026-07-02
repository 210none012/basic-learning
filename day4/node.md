# Day 4: 位置编码、Atlas 论文与 MiniMind 部署

## 第一部分：位置编码

### 一、为什么需要位置编码

Transformer 的 Self-Attention 是**置换等变**的——打乱输入顺序，输出也跟着打乱，这导致模型无法感知 token 的位置信息。位置编码的目标就是给模型注入位置信息。

### 二、绝对位置编码

#### 2.1 可学习位置编码（Learned Positional Encoding）

每个位置对应一个固定向量，直接与词向量相加，所有位置向量都是可训练参数：

$$f_{t \in \{q,k,v\}}(x_i, i) := W_{t \in \{q,k,v\}} (x_i + P_i)$$

注意力分数展开：
$$
\text{Score} = (X_m + P_m)(X_n + P_n)^T = X_m X_n^T + \underbrace{X_m P_n^T + P_m X_n^T}_{\text{噪声}} + P_m P_n^T
$$

**问题**：交叉项 $X_m P_n^T + P_m X_n^T$ 是噪声，需要大量参数来抵消；实际长度超过训练长度时失效。

#### 2.2 正弦位置编码（Sinusoidal Positional Encoding）

通过固定公式直接计算，无参数：

$$
\begin{aligned}
P_{pos, 2t} &= \sin\left(\frac{pos}{10000^{2t / d}}\right) \\
P_{pos, 2t+1} &= \cos\left(\frac{pos}{10000^{2t / d}}\right)
\end{aligned}
$$

**特点**：有一定外推能力（可通过公式计算训练时没见过的位置），但长序列效果仍会退化。

### 三、RoPE（旋转位置编码）

#### 3.1 核心思想

在 Attention 中，token 间相似性用夹角 $\theta$ 表示，越小相似性越高。RoPE 给每个向量加上其所在位置的旋转角度 $m\theta$，使向量间距离变为 $\theta + (n-m)\theta$——相邻 token 的旋转影响小，保持了相对位置关系。

#### 3.2 数学形式

对于第 $i$ 个维度对 $(x_{2i}, x_{2i+1})$，旋转矩阵：
$$
R_{\theta_i, m} = 
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

完整变换（相邻两维一组）：
$$
\text{RoPE}(\mathbf{x}, m) = 
\bigoplus_{i=0}^{d/2-1}
\begin{bmatrix}
x_{2i} \\ x_{2i+1}
\end{bmatrix}
\cdot
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$，$i$ 越小 $\theta_i$ 越大，旋转越快。

#### 3.3 注意力计算

$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{x}_n
$$
$$
\text{Score}(m, n) = \mathbf{q}_m^T \mathbf{k}_n = \mathbf{x}_m^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{x}_n
$$

**关键性质**：$\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{n-m}$，注意力分数只依赖相对位置 $n-m$。

#### 3.4 RoPE 的优势

| 特性 | 说明 |
|------|------|
| 相对位置 | 只依赖位置差，理论上可处理任意长度 |
| 越低的维度转得越快 | 不同转速唯一标识大量向量，避免重叠 |
| 无额外参数 | 旋转矩阵通过公式计算，不增加参数量 |
| 自然衰减 | 远距离 token 间的注意力自然减弱 |

---

## 第二部分：Atlas 论文

> Atlas: Few-shot Learning with Retrieval Augmented Language Models

### 一、架构

| 组件 | 实现 | 说明 |
|------|------|------|
| **检索器** | Contriever（双编码器） | MoCo 对比损失预训练，点积计算相似度 |
| **语言模型** | T5 + Fusion in Decoder | 独立处理每个文档+问题的拼接，再交叉注意力融合 |

### 二、检索器训练：四种损失函数

核心思路：利用**语言模型的监督信号**来训练检索器，让 LM 认为有用的文档排在前面。

| 损失函数 | 监督信号 | 复杂度 | 设计哲学 |
|----------|----------|--------|----------|
| **ADist**（注意力蒸馏） | 交叉注意力分数 | 中 | 模仿 LM 的注意力分布 |
| **EMDR²**（端到端） | 生成概率 | 较高 | 直接优化答案概率 |
| **PDist**（困惑度蒸馏） | 生成概率 | 中 | 预测文档对困惑度的贡献 |
| **LOOP**（留一困惑度） | 留一生成概率 | **最高** | 衡量文档的边际贡献 |

#### ADist（注意力蒸馏）

用 $\alpha_n \cdot \|\mathbf{v}_n\|_2$ 同时考虑关注度和信息量，经 Softmax 得到 LM 侧分布 $p_{\text{attn}}$，最小化其与检索器分布 $p_{\text{retr}}$ 的 KL 散度：
$$
\text{Loss} = \text{KL}(p_{\text{attn}} \parallel p_{\text{retr}})
$$

#### EMDR²（端到端）

将检索文档视为潜在变量，直接最大化生成正确答案的期望概率：
$$
\log \left[ \sum_{k=1}^{K} p_{\text{LM}}(\mathbf{a} \mid \mathbf{q}, \mathbf{d}_k) \cdot p_{\text{retr}}(\mathbf{d}_k \mid \mathbf{q}) \right]
$$

#### PDist（困惑度蒸馏）

训练检索器预测每个文档能在多大程度上降低困惑度：
$$
p_k = \frac{\exp(\log p_{\text{LM}}(\mathbf{a} \mid \mathbf{d}_k, \mathbf{q}))}{\sum_{i=1}^{K} \exp(\log p_{\text{LM}}(\mathbf{a} \mid \mathbf{d}_i, \mathbf{q}))}
$$
$$
\text{Loss} = \text{KL}(p_{\text{LM}} \parallel p_{\text{retr}})
$$

#### LOOP（留一困惑度蒸馏）

移除文档后 LM 性能下降越多，文档越重要：
$$
p_{\text{LOOP}}(\mathbf{d}_k) = \frac{\exp(-\log p_{\text{LM}}(\mathbf{a} \mid \mathcal{D}_K \setminus \{\mathbf{d}_k\}, \mathbf{q}))}{\sum_{i=1}^{K} \exp(-\log p_{\text{LM}}(\mathbf{a} \mid \mathcal{D}_K \setminus \{\mathbf{d}_i\}, \mathbf{q}))}
$$

### 三、自监督预训练任务

#### 前缀语言建模（Prefix LM）

- 文本切分为等长前缀和后续 → 用前缀检索 → 生成后续
- 迫使模型理解长距离依赖

#### 掩码语言建模（MLM）

- 随机遮蔽 15% token（以片段为单位，平均长度 3）
- 被遮蔽部分替换为哨兵 token
- 查询 = 含哨兵的文本，输出 = 哨兵 + 原始片段

### 四、高效的检索器微调

每次更新索引计算量大，三种策略：

| 策略 | 做法 | 开销 | 适用场景 |
|------|------|------|----------|
| **完整索引更新** | 每 R 步重建全部文档向量 | $\frac{N \cdot P_{\text{retr}}}{4BKP_{\text{LM}}R}$ | 大样本 |
| **重排序** | 旧索引取 Top L → 新检索器重编码 Top K | $\frac{L \cdot P_{\text{retr}}}{4KP_{\text{LM}}}$ | 中等 |
| **查询端微调** | 冻结文档编码器，仅训查询编码器 | **零开销** | 少样本 |

**实验发现**：少样本场景下查询端微调效果不降反升（减少过拟合），大样本场景需要完整微调。

---

## 第三部分：MiniMind 部署

### 部署流程

MiniMind 是一个轻量级大语言模型训练项目，用于学习和实践 LLM 的完整训练流程。

```bash
# 克隆仓库、安装依赖
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

python eval_llm.py --load_from ./minimind-3

# 预训练
cd trainer && python train_pretrain.py

# 指令微调
python train_full_sft.py
```
![ping-mu-jie-tu-2026-07-02-111150.png](https://i.postimg.cc/DyJQSMk1/ping-mu-jie-tu-2026-07-02-111150.png)
![ping-mu-jie-tu-2026-07-02-110708.png](https://i.postimg.cc/Gh8P45wP/ping-mu-jie-tu-2026-07-02-110708.png)
![ping-mu-jie-tu-2026-07-02-171344.png](https://i.postimg.cc/RV61WD5d/ping-mu-jie-tu-2026-07-02-171344.png)

---

## 四、Day 4 要点总结

1. **位置编码演进**：可学习 → 正弦 → RoPE，从绝对走向相对，RoPE 已是主流选择
2. **RoPE 核心**：旋转矩阵按位置角度旋转词向量，注意力只依赖相对位置差
3. **Atlas 检索器训练**：用 LM 的信号反向训练检索器，四种损失各有优劣
4. **LOOP 思想**：文档的边际贡献 = 移除后性能下降程度，最直观但计算最贵
5. **高效微调**：少样本用查询端微调（零开销），大样本用完整更新
6. **MiniMind**：完整部署并运行了轻量 LLM 训练环境

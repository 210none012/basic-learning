# Day 7: PyTorch 基础与 LLM 高级主题（推理、编辑、合并、语音）

## 第一部分：PyTorch 基础

### 一、模型保存与加载

```python
torch.save(model.state_dict(), PATH)  # 保存参数
model.load_state_dict(torch.load(PATH))  # 加载参数
```

### 二、GPU 与并行

```python
tensor.to("cuda")           # CPU → GPU
tensor.to("cpu")            # GPU → CPU
torch.get_num_threads()     # 获取 CPU 线程数
torch.set_num_threads(4)    # 设置 CPU 线程数
```

### 三、Autograd 自动求导

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 追踪该 Tensor 的所有操作，用于反向传播 |
| `grad` | 当前 Tensor 的梯度，每次反向传播**累加**，需手动清零 |
| `grad_fn` | 叶子节点为 None，结果节点指示梯度函数类型 |

#### backward() 关键参数

| 参数 | 说明 |
|------|------|
| `tensors` | 需要计算梯度的张量 |
| `grad_tensors` | 矩阵梯度时的权重，shape 与 tensors 相同 |
| `retain_graph` | 保留计算图，允许多次 backward |
| `create_graph` | 构建高阶计算图（用于二阶微分） |

#### torch.autograd.grad()

直接计算指定输入的梯度，比 backward() 更灵活：
- `outputs`：需要梯度的输出
- `inputs`：需要梯度的输入
- `only_inputs`：是否只计算输入梯度
- `allow_unused`：允许某些输入未被使用

### 四、自定义 autograd.Function

继承 `torch.autograd.Function` 实现自定义层：

```python
class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x, b)    # 保存中间结果
        return w * x + b

    @staticmethod
    def backward(ctx, grad_output):
        w, x, b = ctx.saved_tensors       # 取出中间结果
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output
        return grad_w, grad_x, grad_b

# 使用
out = MyLinear.apply(w, x, b)
out.backward(torch.ones_like(out))
```

### 五、nn.Module 体系

| 组件 | 用途 |
|------|------|
| `nn.Module` | 所有神经网络模块的基类 |
| `nn.Parameter` | 可训练参数，自动注册到模型 |
| `nn.ParameterList` | 可变数量参数的容器 |
| `nn.ParameterDict` | 命名参数字典 |
| `nn.Sequential` | 有序模块容器，自动串联 forward |
| `nn.ModuleList` | 模块列表，需手动调 forward |
| `nn.ModuleDict` | 命名模块字典 |

#### nn.Functional vs nn.Module

| | nn.functional | nn.Module |
|------|------|------|
| 状态 | 无状态，无参数 | 有状态，可保存参数 |
| Sequential | ❌ 不能直接组合 | ✅ 可放入 Sequential |
| 推荐场景 | 无参数操作（激活函数等） | 有参数层 |

### 六、模型管理方法

| 方法 | 返回 |
|------|------|
| `model.parameters()` | 所有可训练参数 |
| `model.buffers()` | 非可训练参数（如 BN 的均值/方差） |
| `model.state_dict()` | 参数 + 缓冲区的字典 |
| `model.modules()` | 所有子模块的迭代器 |
| `model.children()` | 直接子模块 |

---

## 第二部分：LLM 推理能力

### 一、核心概念

| 概念 | 含义 |
|------|------|
| **Reasoning** | 模型的思考行为（深度） |
| **Inference** | 模型根据已学知识得到答案 |
| **Test-time Compute** | 测试时的额外计算（脑内剧场） |
| **Test-time Scaling** | 通过深度思考达到训练才能达到的效果 |

### 二、打造推理模型的方法

#### 方式一：不微调参数

**1. CoT（思维链）**

让模型把解题过程拆解成一系列步骤。现在足够长的推理链称为 **Long CoT**（弱模型无法实现）。

**2. 多次采样 + 选择**

| 选择策略 | 做法 |
|------|------|
| **Majority Vote** | 选出现次数最多的答案 |
| **Confidence** | 选模型产生概率最高的答案（CoT decoding） |
| **Best-of-N** | 用 Verifier（语言模型）对 N 个答案打分，选最高分 |

三种部署方式：**Parallel**（并行）、**Sequential**（串行）、**混合**。

**3. Process Verifier（过程验证）**

模型在计算中某一步停下来（如遇到 `</step>`），将不同中间结果交给过程验证器打分，选取每一步中正确概率最大的路径。

#### 方式二：微调参数（类似 Post-Training）

**1. 模仿学习（Imitation Learning）**

- 训练数据格式：`input + reasoning process + ground truth`
- 可以将问题和答案提供给模型，要求写出推理过程，答对的作为训练资料
- 可加入 Verifier 打分
- 也可进行**树状推理**，找出正确路径
- 教会模型在推理出错时**接受反馈并尝试其他路径**

**2. 强化学习（RL）**

- 以结果为导向：只要结果对就是好的
- Foundation Model 的能力很重要
- 会产生**越来越长的推理过程**

#### 四种方法可以组合

后训练用 RL 训练参数，Inference 时再用 Majority Vote 强化。

### 三、推理资源的平衡

#### 过度推理问题

- 推理越长，结果**不一定会越好**
- 解决方案：

| 方案 | 做法 |
|------|------|
| **Chain of Draft** | 限制模型推理长度 |
| 推理树剪枝 | 让模型的推理树小一些 |
| 蒸馏短推理 | 选择老师模型最短的正确推理给学生学习 |
| **implicit CoT** | 每次减少 reasoning 输出量，仍产生正确答案 |
| RL reward | 收集正确答案的平均推理长度，加入 reward 中 |

---

## 第三部分：模型评估的局限性

- 模型答案可能是**记忆出来的** → 更换填空的答案来检验
- 当模型捕捉到出题倾向，基准的效果就**大大降低**（古德哈特定律：一旦目标成为指标，就失去了作用）
- 模型的输出风格也可能影响打分

---

## 第四部分：Model Editing（模型编辑）

### 一、评估标准

| 标准 | 含义 |
|------|------|
| **Reliability** | 目标知识必须被正确修改 |
| **Generalization** | 输出可根据目标改变 |
| **Locality** | 无关输入不受影响 |

Generalization 的评估细分：
- **Paraphrase**：换个说法也正确（"最帅的人→A"）
- **Reverse**：反过来问也成功（"A→最帅的人"）
- **Portability**：建立持续性连接（"最帅的人在哪里工作→"）

### 二、常见方法

#### 不动参数：IKE（In-Context Knowledge Editing）

告诉模型范例后提出要求——本质是**关闭了 RAG** 的上下文学习。

#### 改变参数

**1. ROME（Rank-One Model Editing）**

- 人类决定如何编辑
- 找出神经网络中与编辑知识最相关的部分，修改该部分参数
- 寻找方法：原问题得到内部向量 → 盖住部分问题 → 将原向量替换到相同位置 → 如果输出正确答案说明关联性大
- 向向量中加入目标向量完成修改

**2. 超网络编辑**

- 另一 AI 决定如何编辑
- 将待编辑模型的参数 $a$、问题、答案输入编辑模型
- 编辑模型输出与 $a$ 相同规模的参数 $e$
- $e$ 加到待编辑模型上完成修改
- 训练时需要加上保持其他问题答案不变的例子

---

## 第五部分：Model Merging（模型合并）

- 同一基模型训练出的两个模型可通过**参数加减**合并获得新能力
- 同样减去向量也可以**失去对应能力**
- 每个模型是节点，获得能力的过程是向量，通过向量加减实现能力转移
- 关键前提：两个模型**修改的参数不同**时才容易成功

---

## 第六部分：Speech LLM（语音大模型）

### 一、语音 Token 化

#### 两种 Token 类型

| 类型 | 方法 | 处理 |
|------|------|------|
| **Semantic Token** | Speech SSL Model 提取向量 → 离散化 | Duplicate 去重 → BPE 合并 |
| **Acoustic Token** | Neural Codec（RAQ） | 多组 token 表示不同信息层次 |

#### Token 质量评估

| 方法 | 方式 |
|------|------|
| **Codec-SUPERB** | 语音→token→还原，比较质量损失 |
| **DASB** | 声音→token，直接训练检测质量 |

### 二、Decoding 策略

#### 粗细粒度解码

| 策略 | 做法 | 局限 |
|------|------|------|
| 粗→细排列 | 从粗 token 到细粒度 token 逐层生成 | 无法流式 |
| 两阶段 LLM | LLM1 产粗粒度 → LLM2 以粗为输入产细粒度 | — |
| 同位置排列 | 不同粒度的同一 token 放一起直接输出声音 | — |
| **Acoustic Delay** | 每次生成已有粗粒度 token 的细粒度 token | — |

架构：分为 **Temporal Transformer** 与 **Depth Transformer**。

#### Discrete Token

生成时给定输入可能产生不同输出，使用 discrete 使模型**只选一个**答案，而非产生平均。

### 三、语音-文字联合训练

- 纯语音信号有效内容有限，难以训练出良好效果
- **文字模型本身包含文本和语义** → 以文字模型为基础训练语音模型
- 主流思路：同时产生文字和语音 token
  - 文字生成完后用特定符号替代
  - 每个文字 token 和语音 token 保证指定长度
  - 有模型预测每个文字对应多少语音 token

**核心难题**：语音 token 数量远超文字 token，如何对齐？

### 四、TTS（文本转语音）

可同时接受文字与语音信号，学习输出对应的语气。

---

## 第七部分：Embedding 相似度

| 度量 | 含义 | 场景 |
|------|------|------|
| **余弦相似度** | 语义的相近程度（是否在同一方向） | 语义检索 |
| **欧氏距离** | 整体是否相似（空间距离） | 整体匹配 |

---

## 八、要点总结

1. **PyTorch 核心**：Autograd 链式求导、自定义 Function、nn.Module 模块体系
2. **推理增强**：CoT + Majority Vote + Best-of-N + Process Verifier 可组合使用
3. **RL 推理**：以结果为导向，但推理过程越长不意味着越好
4. **Model Editing**：ROME 定位关联参数 → 修改参数；IKE 不动参数
5. **Model Merging**：参数加减实现能力转移，前提是修改参数不重叠
6. **Speech LLM**：Semantic + Acoustic 双 token，Acoustic Delay 解决粒度对齐
7. **余弦相似度 vs 欧氏距离**：语义方向 vs 整体相似度

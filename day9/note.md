# Day 9: BERT 微调实战与 Transformer 分词详解

## 第一部分：项目架构

基于 BERT 的中文情感二分类（正向/负向），完整微调流水线：

```
transformer_fine_tune/
├── Mydata.py    → 数据集加载
├── net.py       → 模型定义（BERT + 分类头）
├── trainer.py   → 训练循环
├── test.py      → 批量评估
└── run.py       → 单条推理/交互测试
```

---

## 第二部分：Tokenizer 分词器

### 一、batch_encode_plus 参数

```python
token = BertTokenizer.from_pretrained("bert-base-chinese")
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[句子1, 句子2],
    add_special_tokens=True,        # 自动加 [CLS] [SEP]
    max_length=30,                  # 最大长度
    truncation=True,                # 超长截断
    padding="max_length",           # 不足补 [PAD]
    return_tensors=None,            # pt=返回PyTorch张量
    return_attention_mask=True,     # 返回注意力掩码
    return_length=True,             # 返回实际长度
)
```

### 二、编码输出字段

| 字段 | 含义 |
|------|------|
| `input_ids` | 编码后的 token ID |
| `token_type_ids` | 第一句和特殊符号为 0，第二句为 1 |
| `attention_mask` | PAD 位置为 0，其余为 1 |
| `special_tokens_mask` | 特殊符号位置为 1 |
| `length` | 实际 token 长度 |

### 三、词表操作

```python
# 编码 + 解码
token.encode("阳光照在大地上", max_length=10, truncation=True)
token.decode(ids)

# 获取词表
vocab = token.get_vocab()

# 添加新词
token.add_tokens(new_tokens=["阳光", "大地"])

# 添加特殊符号
token.add_special_tokens({"eos_token": "[EOS]"})
```

> 新词添加到词典最后，添加后需重新 `get_vocab()`。

---

## 第三部分：模型定义（net.py）

### 架构

```python
pretrained = BertModel.from_pretrained("bert-base-chinese")  # 冻结

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)  # BERT hidden_size=768 → 2分类

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():  # BERT 只前向，不参与反向训练
            out = pretrained(input_ids, attention_mask, token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])  # 取 [CLS] 向量
        out = out.softmax(dim=1)
        return out
```

### 关键设计

| 设计 | 说明 |
|------|------|
| `torch.no_grad()` | 冻结 BERT，只训练分类头（参数高效） |
| `last_hidden_state[:, 0]` | 取 `[CLS]` token 的表示用于分类 |
| `Linear(768, 2)` | 768 维 → 2 分类（正向/负向） |

---

## 第四部分：数据集（Mydata.py）

### 自定义 Dataset

```python
from torch.utils.data import Dataset
from datasets import load_dataset

class Mydataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("lansinuote/ChnSentiCorp")
        self.dataset = self.dataset[split]  # train/validation/test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]["text"], self.dataset[item]["label"]
```

### 三个必须方法

| 方法 | 作用 |
|------|------|
| `__init__` | 初始化，加载数据 |
| `__len__` | 返回数据集大小 |
| `__getitem__` | 返回单个样本 (text, label) |

---

## 第五部分：collate_fn 批处理

```python
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=500,
        return_tensors="pt",
    )

    return (data["input_ids"], data["attention_mask"],
            data["token_type_ids"], torch.LongTensor(labels))
```

DataLoader 调用：
```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,          # 丢弃不完整批次
    collate_fn=collate_fn    # 自定义批处理
)
```

---

## 第六部分：训练循环（trainer.py）

```python
model = Model().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=5e-4)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for input_ids, attention_mask, token_type_ids, labels in train_loader:
        # 数据移到 GPU
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向 → 损失 → 反向 → 优化
        out = model(input_ids, attention_mask, token_type_ids)
        loss = loss_func(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
```

### 训练核心三行

| 步骤 | 代码 | 作用 |
|------|------|------|
| 清零 | `optimizer.zero_grad()` | 防止梯度累加 |
| 反向 | `loss.backward()` | 计算当前梯度 |
| 更新 | `optimizer.step()` | 根据梯度更新参数 |

### 关键配置

| 参数 | 值 | 说明 |
|------|------|------|
| Optimizer | `AdamW` | Adam + 权重衰减正则化 |
| Loss | `CrossEntropyLoss` | 多分类交叉熵（自带 Softmax） |
| Epoch | `100` | 训练轮数 |
| Batch Size | `32` | 每批样本数 |
| Max Length | `500` | 输入最大 token 数 |

---

## 第七部分：评估与推理

### 批量评估（test.py）

```python
model.eval()
with torch.no_grad():
    out = model(input_ids, attention_mask, token_type_ids)
    out = out.argmax(dim=1)  # 取概率最大的类别
    acc += (out == labels).sum().item()
print(acc / total)
```

### 单条推理（run.py）

```python
model.load_state_dict(torch.load("params/13bert.pt"))
model.eval()

while True:
    text = input("请输入：")
    input_ids, attention_mask, token_type_ids = collate_fn(text)
    with torch.no_grad():
        out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
    print(names[out])  # "负向评价" / "正向评价"
```

### 推理关键

| 操作 | 说明 |
|------|------|
| `model.eval()` | 评估模式（关闭 Dropout） |
| `torch.no_grad()` | 不计算梯度，减少显存 |
| `argmax(dim=1)` | 取概率最大的类别索引 |

---

## 八、完整流水线总结

```
数据加载 → Tokenizer编码 → DataLoader批处理
    ↓
BERT(冻结) → [CLS]向量 → Linear(768→2) → Softmax
    ↓
CrossEntropyLoss → AdamW更新参数
    ↓
每epoch保存 → 评估/推理
```

## 九、要点总结

1. **Tokenizer**：`batch_encode_plus` 批量编码，`add_tokens` 扩展词表
2. **模型设计**：冻结 BERT + 训练分类头，`[CLS]` 向量做分类
3. **Dataset**：三个必须方法 `__init__`/`__len__`/`__getitem__`
4. **collate_fn**：自定义批处理 → 编码 + 转 Tensor
5. **训练三连**：`zero_grad()` → `backward()` → `step()`
6. **推理模式**：`eval()` + `no_grad()` + `argmax()`

---

## 第十部分：补充知识

### 一、Unknown Token 处理

当模型遇到词表中不存在的词时，会编码为 `[UNK]`（unknown token）。

解决方案：

```python
token.add_tokens(new_tokens=["新词1", "新词2"])
token.add_special_tokens({"eos_token": "[EOS]"})
```

> 添加新词后需重新获取词表：`vocab = token.get_vocab()`

### 二、Visdom：PyTorch 可视化工具

```bash
pip install visdom
python -m visdom.server     # 启动服务，浏览器访问 localhost:8097
```

用于实时监控训练过程中的 loss 曲线、准确率等指标。

### 三、交叉验证（Cross Validation）

| 方法 | 做法 | 适用场景 |
|------|------|----------|
| **简单交叉验证** | 划分为训练集 + 测试集 | 数据充足，快速验证 |
| **2-折交叉验证** | A 训 B 测 → B 训 A 测 | 数据较少时 |
| **K-折交叉验证** | 分 K 份，轮换 K-1 份训练、1 份测试 | 最常用，K 通常取 5 或 10 |
| **留一交叉验证** | 保留 1 份样本为测试，其余训练 | 数据极度稀缺 |

#### K-折交叉验证流程

```
数据集 → 分成 K 折
  第1轮: 折2..K训练 → 折1测试
  第2轮: 折1+3..K训练 → 折2测试
  ...
  第K轮: 折1..K-1训练 → 折K测试
最终得分 = K轮结果的平均值
```

> 交叉验证能更充分利用有限数据，同时评估模型的稳定性。

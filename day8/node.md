# Day 8: argparse与Transformers 

## 第一部分：argparse 命令行参数

### 一、为什么用 argparse

不用 argparse 的问题：
- 修改超参数需要找到代码行修改，繁琐易错
- 传参时需写明所有参数，代码冗长

argparse 的优势：
- 运行时直接指定超参数
- 所有参数储存在 `args` 对象中，传参只需传 `args`

### 二、基本用法

```python
import argparse

parser = argparse.ArgumentParser(description="描述信息")
parser.add_argument("food", help="位置参数说明")
parser.add_argument("-n", "--name", type=str, required=True, help="可选参数")
parser.add_argument("--amp", action="store_true", help="布尔标志")

args = parser.parse_args()
print(args.food, args.name, args.amp)
```
![ping-mu-jie-tu-2026-07-08-164749.png](https://i.postimg.cc/VvyRbyZG/ping-mu-jie-tu-2026-07-08-164749.png)
![ping-mu-jie-tu-2026-07-08-165008.png](https://i.postimg.cc/jj5vp4Jm/ping-mu-jie-tu-2026-07-08-165008.png)
### 三、参数类型

| 类型 | 写法 | 特点 |
|------|------|------|
| **位置参数** | `"food"`（无 `-`） | 按顺序传值，不用写名称 |
| **可选参数** | `-n` / `--name` | 可选，设 `required=True` 后必填 |

### 四、add_argument 关键参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `help` | 参数说明（`-h` 可查看） | `help="请输入姓名"` |
| `type` | 指定参数类型 | `type=int` |
| `required` | 是否必填 | `required=True` |
| `default` | 默认值 | `default="18"` |
| `choices` | 限定可选值 | `choices=[1, 2]` |
| `action` | 不指定值的动作 | `store_true` / `store_false` |
| `dest` | 代码中的变量名（不能用命令行指定） | `dest="na"` → `args.na` |

> `-h` 或 `--help` 查看所有参数的帮助信息。不要同时使用 `action` 和 `default`。

---

## 第二部分：Transformers 生态

| 库 | 功能 |
|------|------|
| **Transformers** | 核心库：模型加载、训练、Pipeline |
| **Tokenizer** | 分词器：文本 ↔ token 序列互转 |
| **Datasets** | 数据集库：数据集的加载与处理 |
| **Evaluate** | 评估函数：各种评价指标的计算 |
| **PEFT** | 高效微调：LoRA 等参数高效微调方法 |
| **Accelerate** | 分布式训练：大模型加载与推理方案 |
| **Optimum** | 优化加速：支持 ONNX、OpenVINO 等后端 |
| **Gradio** | 可视化部署：几行代码实现 Web 交互演示 |

---

## 第三部分：模型调用实战

### 一、使用网络 API 调用模型

```python
import requests

# 匿名访问
response = requests.post(API_URL, json={"inputs": "输入文本"})

# 使用 Token 访问
API_KEY = "密钥"
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.post(API_URL, headers=headers, json={"inputs": "输入"})
```

### 二、下载模型到本地

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "本地保存路径"

AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
```
![ping-mu-jie-tu-2026-07-08-204920.png](https://i.postimg.cc/YSqypwTf/ping-mu-jie-tu-2026-07-08-204920.png)

| 类 | 用途 |
|------|------|
| `AutoModelForCausalLM` | 因果语言模型（GPT 系列） |
| `AutoTokenizer` | 自动匹配分词器 |

### 三、加载本地模型 + Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("模型路径")
tokenizer = AutoTokenizer.from_pretrained("模型路径")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

output = generator("你好，我是", max_length=50, num_return_sequences=1)
```
![ping-mu-jie-tu-2026-07-08-210858.png](https://i.postimg.cc/nhK1jwqT/ping-mu-jie-tu-2026-07-08-210858.png)
### 四、生成参数详解

| 参数 | 作用 | 说明 |
|------|------|------|
| `max_length` | 最大输出 token 数 | 控制生成长度 |
| `num_return_sequences` | 返回几个独立序列 | >1 时分多段回答 |
| `truncation` | 是否截断输入 | 适配 max_length |
| `temperature` | 随机性控制 | 越低越保守，越高越多样 |
| `top_k` | 只从概率最高的 k 个词中选择 | 避免概率极低的词 |
| `top_p` | 累计概率阈值采样 | 词概率累计达 p 即停止 |
| `clean_up_tokenization_spaces` | 清理生成文本空格 | — |

#### 采样策略对比

| 策略 | 原理 | 效果 |
|------|------|------|
| **Temperature** | 缩放概率分布 | T<1 更确定，T>1 更多样 |
| **Top-K** | 限制候选词池为前 K 个 | 过滤极低概率词 |
| **Top-P（Nucleus）** | 选累计概率达 p 的最小词集 | 动态适应分布 |
| **组合使用** | Top-K + Top-P + Temperature | 平衡质量与多样性 |

### 五、模型加载方式总结

| 方式 | 代码 | 适用 |
|------|------|------|
| API 调用 | `requests.post()` | 快速原型，无需本地资源 |
| Pipeline | `pipeline("text-generation", ...)` | 最简方式，三行搞定 |
| 手动加载 | `from_pretrained()` + 手动 forward | 最大灵活性 |

---

## 四、要点总结

1. **argparse** 实现命令行参数管理：位置参数/可选参数、`action`、`dest`、`choices`
2. **Transformers 全家桶**：PEFT 微调、Accelerate 分布式、Gradio 部署
3. **模型加载**：本地下载用 `cache_dir`，加载用 `from_pretrained()`
4. **Pipeline 一行生成**：`pipeline("text-generation", model=model, tokenizer=tokenizer)`
5. **采样策略**：Temperature + Top-K + Top-P 组合控制生成质量与多样性

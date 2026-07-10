# Day 10: Haystack 实战与 RAG 优化方案

## 第一部分：Haystack 框架基础

### 一、Haystack 

Haystack 是一个用于构建 RAG 和 NLP Pipeline 的框架，提供模块化组件：

| 组件类型 | 常用类 |
|----------|--------|
| Document Store | `InMemoryDocumentStore` |
| Embedder | `SentenceTransformersDocumentEmbedder` |
| Retriever | `InMemoryEmbeddingRetriever`, `InMemoryBM25Retriever` |
| Generator | `OpenAIChatGenerator` |
| Router | `TransformersTextRouter` |

### 二、索引 Pipeline（indexing.py）

#### 核心流程

```
文档 → Cleaner(清洗) → Splitter(切割) → Embedder(向量化) → Writer(入库)
```

```python
def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="period", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(model="thenlper/gte-large")
    document_writer = DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)

    pipeline = Pipeline()
    pipeline.add_component("cleaner", document_cleaner)
    pipeline.add_component("splitter", document_splitter)
    pipeline.add_component("embedder", document_embedder)
    pipeline.add_component("writer", document_writer)

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    return pipeline
```

#### 关键配置

| 参数 | 说明 |
|------|------|
| `split_by="period"` | 按句号分割 |
| `split_length=2` | 每块 2 句话 |
| `DuplicatePolicy.OVERWRITE` | 重复文档覆盖写入 |
| `metadata_fields_to_embed` | 嵌入元数据字段增强检索 |

#### 检索对比

```python
retrieval_pipeline.connect("text_embedder", "retriever")                 # 无元数据
retrieval_pipeline.connect("text_embedder", "retriever_with_embeddings") # 带标题嵌入
```

> 将 `metadata_fields_to_embed=["title"]` 可使检索时同时匹配标题语义。

### 三、Router 路由（router.py）

#### 1. TransformersTextRouter

将查询分为 KEYWORD 和 QUESTION/STATEMENT 两类，分别走不同检索策略：

```python
text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
# LABEL_0 → Keyword Query → BM25 稀疏检索
# LABEL_1 → Question → Embedding 密集检索
```
![ping-mu-jie-tu-2026-07-10-153332.png](https://i.postimg.cc/NMvvJdp1/ping-mu-jie-tu-2026-07-10-153332.png)
#### 2. TransformersZeroShotTextRouter

零样本分类路由：

```python
text_router = TransformersZeroShotTextRouter(labels=["music", "cinema"])
text_router.warm_up()
result = text_router.run(text="What is the Rolling Stones first album?")
# → "music"
```
![ping-mu-jie-tu-2026-07-10-160333.png](https://i.postimg.cc/W30TTBdc/ping-mu-jie-tu-2026-07-10-160333.png)
#### 3. 路由 Pipeline 示例

```python
pipeline.connect("text_router.LABEL_0", "text_embedder")        # Keyword → 密集
pipeline.connect("text_embedder", "embedding_retriever")
pipeline.connect("text_router.LABEL_1", "bm25_retriever")       # Question → BM25
```

#### 4. Router + Reader 组合

```python
pipeline.connect("text_router.LABEL_0", "bm25_retriever_0")     # Statement → 检索
pipeline.connect("bm25_retriever_0", "reader")                   # → 抽取答案
pipeline.connect("text_router.LABEL_1", "bm25_retriever_1")     # Question → 仅检索
```
![ping-mu-jie-tu-2026-07-10-161632.png](https://i.postimg.cc/TwbYhyHr/ping-mu-jie-tu-2026-07-10-161632.png)


---

## 第四部分：完整 RAG Pipeline（basic.py）

### 一、自定义 Embedder

不使用 SentenceTransformers，手写 Transformers 嵌入：

```python
class TransformersEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean Pooling
        return embeddings.numpy()
```

### 二、自定义 Haystack 组件

```python
@component
class TextEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        embedding = self.embedder.encode([text])[0].tolist()
        return {"embedding": embedding}
```

### 三、完整 RAG 流水线

```python
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
```
![ping-mu-jie-tu-2026-07-10-111321.png](https://i.postimg.cc/zX2ZZzxh/ping-mu-jie-tu-2026-07-10-111321.png)
#### 执行流程

```
查询文本 → Embedder → Retriever → Prompt Builder → LLM → 回答
```

---

## 第五部分：16 种 RAG 优化方案

| #   | 方案                         | 核心思想                          |
| --- | -------------------------- | ----------------------------- |
| 1   | Simple RAG                 | 固定长度切块 + 向量检索                 |
| 2   | Semantic Chunking          | 基于语义切块，保持句子完整性                |
| 3   | Context Enriched Retrieval | 检索时返回目标块及**相邻块**              |
| 4   | Contextual Chunk Headers   | 为每块添加**简洁标题**，增强可检索性          |
| 5   | Document Augmentation      | 生成 QA 对或潜在查询作为额外检索入口          |
| 6   | Query Transformation       | LLM 重写/扩展用户查询                 |
| 7   | **Reranker**               | 初检后用精细模型重新打分排序                |
| 8   | RSE（语义扩展重排序）               | 向量检索 + 上下文窗口加权 + 语义扩展         |
| 9   | Feedback Loop              | 用户反馈驱动系统自我进化                  |
| 10  | **Adaptive RAG**           | 根据问题类型选择**不同策略**              |
| 11  | Self-RAG                   | LLM 自判断是否需要检索 + 相关性评估         |
| 12  | Knowledge Graph            | 实体-关系-属性三元组，高成本高精度            |
| 13  | Hierarchical Indices       | 小块匹配关键词 → 大块提供完整语义            |
| 14  | HyDE                       | LLM 生成假设答案，用其向量代替原始查询         |
| 15  | **Fusion**                 | 多种检索结果**打分加权融合**              |
| 16  | **CRAG**                   | 按相关性分级处理（低→Web / 中→混合 / 高→直接） |


---

## 六、要点总结

1. **Haystack Pipeline**：模块化串联 Cleaner → Splitter → Embedder → Retriever → Generator
2. **Router**：TextRouter 分类查询类型，ZeroShotRouter 无需微调即可分类
3. **自定义组件**：`@component` 装饰器 + `@component.output_types` 定义输出类型
4. **索引增强**：`metadata_fields_to_embed` 将元数据嵌入向量，提升检索准确度
5. **RAG 优化 Top 3**：Adaptive RAG（自适应）、Fusion（融合）、CRAG（分级纠正）

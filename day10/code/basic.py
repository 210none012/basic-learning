import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import component
from haystack.utils import Secret
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch

# ============ 1. 自定义嵌入器 ============
class TransformersEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()

# ============ 2. 加载数据 ============
print("加载数据...")
document_store = InMemoryDocumentStore()
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# ============ 3. 生成嵌入 ============
print("生成文档嵌入...")
embedder = TransformersEmbedder()
texts = [doc.content for doc in documents]
embeddings = embedder.encode(texts)

for doc, emb in zip(documents, embeddings):
    doc.embedding = emb

document_store.write_documents(documents)
print(f"索引了 {document_store.count_documents()} 个文档")

# ============ 4. 查询组件 ============
@component
class TextEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder
    
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        embedding = self.embedder.encode([text])[0].tolist()
        return {"embedding": embedding}

# ============ 5. 构建管道 ============
text_embedder = TextEmbedder(embedder)
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

template = [
    ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")
]
prompt_builder = ChatPromptBuilder(template=template, required_variables="*")

chat_generator = OpenAIChatGenerator(
    model="deepseek-chat",
    api_key=Secret.from_token("key"),
    api_base_url="https://api.deepseek.com/v1"
)

basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# ============ 6. 查询 ============
question = "What does Rhodes Statue look like?"
print(f"\n📝 问题: {question}")

response = basic_rag_pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"question": question}
})

print(f"💡 回答: {response['llm']['replies'][0].text}")
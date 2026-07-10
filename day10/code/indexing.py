from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice

#indexing:将文档存入库
def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="period", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large", meta_fields_to_embed=metadata_fields_to_embed
    )
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline


import wikipedia
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

wikipedia.set_user_agent("haystack-tutorials")

some_bands = """The Beatles,The Cure""".split(",")

raw_docs = []

for title in some_bands:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url": page.url})
    raw_docs.append(doc)

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_store_with_embedded_metadata = InMemoryDocumentStore(embedding_similarity_function="cosine")

indexing_pipeline = create_indexing_pipeline(document_store=document_store)
indexing_with_metadata_pipeline = create_indexing_pipeline(
    document_store=document_store_with_embedded_metadata, metadata_fields_to_embed=["title"]
)

indexing_pipeline.run({"cleaner": {"documents": raw_docs}})
indexing_with_metadata_pipeline.run({"cleaner": {"documents": raw_docs}})
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large"))
retrieval_pipeline.add_component(
    "retriever", InMemoryEmbeddingRetriever(document_store=document_store, scale_score=False, top_k=3)
)
retrieval_pipeline.add_component(
    "retriever_with_embeddings",
    InMemoryEmbeddingRetriever(document_store=document_store_with_embedded_metadata, scale_score=False, top_k=3),
)

retrieval_pipeline.connect("text_embedder", "retriever")
retrieval_pipeline.connect("text_embedder", "retriever_with_embeddings")
result = retrieval_pipeline.run({"text_embedder": {"text": "Have the Beatles ever been to Bangor?"}})

print("Retriever Results:\n")
for doc in result["retriever"]["documents"]:
    print(doc)

print("Retriever with Embeddings Results:\n")
for doc in result["retriever_with_embeddings"]["documents"]:
    print(doc)

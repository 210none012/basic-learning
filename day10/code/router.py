# from haystack_integrations.components.routers.transformers import TransformersTextRouter
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
# text_router.warm_up()
# queries = [
#     "Arya Stark father",  # Keyword Query
#     "Who was the father of Arya Stark",  # Interrogative Query
#     "Lord Eddard was the father of Arya Stark",  # Statement Query
# ]
# result = text_router.run(text=queries[0])
# print(next(iter(result)))
# import pandas as pd

# results = {"Query": [], "Output Branch": [], "Class": []}

# for query in queries:
#     result = text_router.run(text=query)
#     results["Query"].append(query)
#     results["Output Branch"].append(next(iter(result)))
#     results["Class"].append("Keyword Query" if next(iter(result)) == "LABEL_0" else "Question/Statement")

# print(pd.DataFrame.from_dict(results))

# queries = [
#     "Who was the father of Arya Stark",  # Interrogative Query
#     "Lord Eddard was the father of Arya Stark",  # Statement Query
# ]

# results = {"Query": [], "Output Branch": [], "Class": []}

# for query in queries:
#     result = text_router.run(text=query)
#     results["Query"].append(query)
#     results["Output Branch"].append(next(iter(result)))
#     results["Class"].append("Question" if next(iter(result)) == "LABEL_1" else "Statement")

# print(pd.DataFrame.from_dict(results))

########################test2############################

# from haystack_integrations.components.routers.transformers import TransformersZeroShotTextRouter
# import pandas as pd

# text_router = TransformersZeroShotTextRouter(labels=["music", "cinema"])
# text_router.warm_up()
# queries = [
#     "In which films does John Travolta appear?",  # cinema
#     "What is the Rolling Stones first album?",  # music
#     "Who was Sergio Leone?",  # cinema
# ]
# sent_results = {"Query": [], "Output Branch": []}

# for query in queries:
#     result = text_router.run(text=query)
#     sent_results["Query"].append(query)
#     sent_results["Output Branch"].append(next(iter(result)))

# print(pd.DataFrame.from_dict(sent_results))

# text_router = TransformersZeroShotTextRouter(labels=["Game of Thrones", "Star Wars", "Lord of the Rings"])
# text_router.warm_up()

# queries = [
#     "Who was the father of Arya Stark",  # Game of Thrones
#     "Who was the father of Luke Skywalker",  # Star Wars
#     "Who was the father of Frodo Baggins",  # Lord of the Rings
# ]

# results = {"Query": [], "Output Branch": []}

# for query in queries:
#     result = text_router.run(text=query)
#     results["Query"].append(query)
#     results["Output Branch"].append(next(iter(result)))

# print(pd.DataFrame.from_dict(results))

####################test3##########################

from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack import Pipeline
from haystack_integrations.components.routers.transformers import TransformersTextRouter

document_store = InMemoryDocumentStore()
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)

query_classification_pipeline = Pipeline()
query_classification_pipeline.add_component("text_router", text_router)
query_classification_pipeline.add_component("text_embedder", text_embedder)
query_classification_pipeline.add_component("embedding_retriever", embedding_retriever)
query_classification_pipeline.add_component("bm25_retriever", bm25_retriever)

query_classification_pipeline.connect("text_router.LABEL_0", "text_embedder")
query_classification_pipeline.connect("text_embedder", "embedding_retriever")
query_classification_pipeline.connect("text_router.LABEL_1", "bm25_retriever")

# Useful for framing headers
equal_line = "=" * 30
print(equal_line)

# Run only the dense retriever on the full sentence query
res_1 = query_classification_pipeline.run({"text_router": {"text": "Who is the father of Arya Stark?"}})
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print(res_1)

# Run only the sparse retriever on a keyword based query
res_2 = query_classification_pipeline.run({"text_router": {"text": "arya stark father"}})
print(f"\n\n{equal_line}\nKEYWORD QUERY RESULTS\n{equal_line}")
print(res_2)

#######################test4#####################

from haystack_integrations.components.readers.transformers import TransformersExtractiveReader

query_classification_pipeline = Pipeline()
query_classification_pipeline.add_component("bm25_retriever_0", InMemoryBM25Retriever(document_store))
query_classification_pipeline.add_component("bm25_retriever_1", InMemoryBM25Retriever(document_store))
query_classification_pipeline.add_component(
    "text_router", TransformersTextRouter(model="shahrukhx01/question-vs-statement-classifier")
)
query_classification_pipeline.add_component("reader", TransformersExtractiveReader())

query_classification_pipeline.connect("text_router.LABEL_0", "bm25_retriever_0")
query_classification_pipeline.connect("bm25_retriever_0", "reader")
query_classification_pipeline.connect("text_router.LABEL_1", "bm25_retriever_1")

# Useful for framing headers
equal_line = "=" * 30
print(equal_line)

# Run the retriever + reader on the question query
query = "Who is the father of Arya Stark?"
res_1 = query_classification_pipeline.run({"text_router": {"text": query}, "reader": {"query": query}})
print(f"\n\n{equal_line}\nQUESTION QUERY RESULTS\n{equal_line}")
print(res_1)

# Run only the retriever on the statement query
query = "Arya Stark was the daughter of a Lord"
res_2 = query_classification_pipeline.run({"text_router": {"text": query}, "reader": {"query": query}})
print(f"\n\n{equal_line}\nKEYWORD QUERY RESULTS\n{equal_line}")
print(res_2)


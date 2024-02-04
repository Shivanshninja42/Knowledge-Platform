import nest_asyncio

nest_asyncio.apply()

import openai

import os


os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.retrievers import BM25Retriever
from llama_index.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.llms import OpenAI
documents = SimpleDirectoryReader("data/paul_graham").load_data()


llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    service_context=service_context,
)
retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
from llama_index.response.notebook_utils import display_source_node


nodes = retriever.retrieve("What happened at Viaweb and Interleaf?")
for node in nodes:
    display_source_node(node)

nodes = retriever.retrieve("What did Paul Graham do after RISD?")
for node in nodes:
    display_source_node(node)

from llama_index.tools import RetrieverTool

vector_retriever = VectorIndexRetriever(index)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

retriever_tools = [
    RetrieverTool.from_defaults(
        retriever=vector_retriever,
        description="Useful in most cases",
    ),
    RetrieverTool.from_defaults(
        retriever=bm25_retriever,
        description="Useful if searching about specific information",
    ),
]

from llama_index.retrievers import RouterRetriever

retriever = RouterRetriever.from_defaults(
    retriever_tools=retriever_tools,
    service_context=service_context,
    select_multi=True,
)


nodes = retriever.retrieve(
    "Can you give me all the context regarding the author's life?"
)
for node in nodes:
    display_source_node(node)

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI


documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(chunk_size=256, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
index = VectorStoreIndex(
    nodes, storage_context=storage_context, service_context=service_context
)

from llama_index.retrievers import BM25Retriever


vector_retriever = index.as_retriever(similarity_top_k=10)


bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)


from llama_index.retrievers import BaseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)


        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

index.as_retriever(similarity_top_k=5)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

from llama_index.indices.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

from llama_index import QueryBundle

nodes = hybrid_retriever.retrieve(
    "What is the impact of climate change on the ocean?"
)
reranked_nodes = reranker.postprocess_nodes(
    nodes,
    query_bundle=QueryBundle(
        "What is the impact of climate change on the ocean?"
    ),
)

print("Initial retrieval: ", len(nodes), " nodes")
print("Re-ranked retrieval: ", len(reranked_nodes), " nodes")

from llama_index.response.notebook_utils import display_source_node

for node in reranked_nodes:
    display_source_node(node)


from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    service_context=service_context,
)

response = query_engine.query(
    "What is the impact of climate change on the ocean?"
)
from llama_index.response.notebook_utils import display_response

display_response(response)


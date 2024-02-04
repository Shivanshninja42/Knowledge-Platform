import logging
import sys
import openai
import os

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import OpenAI
from IPython.display import Markdown, display

gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)

gpt4 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)

from llama_index.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)


step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)


step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    llm=gpt3, verbose=True
)
index_summary = "Used to answer questions about the author"
from llama_index.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)

query_engine = index.as_query_engine(service_context=service_context_gpt4)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary,
)
response_gpt4 = query_engine.query(
    "Who was in the first batch of the accelerator program the author"
    " started?",
)
display(Markdown(f"<b>{response_gpt4}</b>"))

sub_qa = response_gpt4.metadata["sub_qa"]
tuples = [(t[0], t[1].response) for t in sub_qa]
print(tuples)

response_gpt4 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)
print(response_gpt4)


query_engine = index.as_query_engine(service_context=service_context_gpt3)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary,
)

response_gpt3 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)
print(response_gpt3)

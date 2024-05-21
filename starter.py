# import os
# from dotenv import load_dotenv
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# load_dotenv()
 
 
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)

# query_engine = index.as_query_engine()
# response = query_engine.query("what he is work before college?")
# print(response);

import os
from dotenv import load_dotenv
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    RAKEKeywordTableIndex,
)


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from llama_index.llms.openai import OpenAI

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")


reader = SimpleDirectoryReader(input_files=["./data/lyft_2021.pdf"])
data = reader.load_data()

index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

response = query_engine.query(
    "Which are Risks Related to General Economic Factors?"
    "page reference after each statement."
)
response.print_response_stream()
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from starlette.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from pymilvus import MilvusClient

# 加载环境变量，读取本地 .env 文件，里面定义了 OPENAI_API_KEY
_ = load_dotenv(find_dotenv())

# llm
llm = ChatOpenAI(model="qwen3:1.7b", temperature=0)

# 加载文档,可换成PDF、txt、doc等其他格式文档
loader = TextLoader('../docs/解答手册.md', encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_language(language="markdown", chunk_size=200, chunk_overlap=0)
pages = text_splitter.split_documents(documents)

# 加载PDF
# loader = PyMuPDFLoader("../docs/解答手册.pdf")
# pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.create_documents(
    [page.page_content for page in pages]
)


print(texts)

import ollama
def emb_text(text):
    response = ollama.Client(host='http://47.98.197.208:11434').embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])






milvus_client = MilvusClient(uri="http://47.98.197.208:19530")
print(milvus_client)


collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)


milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
)













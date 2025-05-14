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

# 选择向量模型，并灌库
db = FAISS.from_documents(texts, OpenAIEmbeddings(model="nomic-embed-text",base_url="http://47.98.197.208:11434"))
# 获取检索器，选择 top-2 相关的检索结果
retriever = db.as_retriever(search_kwargs={"k": 1})

# 创建带有 system 消息的模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个对接问题排查机器人。
               你的任务是根据下述给定的已知信息回答用户问题。
               确保你的回复完全依据下述已知信息，不要编造答案。
               请用中文回答用户问题。

               已知信息:
               {context} """),
    ("user", "{question}")
])

# 自定义的提示词参数
chain_type_kwargs = {
    "prompt": prompt_template,
}

# 定义RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 使用stuff模式将上下文拼接到提示词中
    chain_type_kwargs=chain_type_kwargs,
    retriever=retriever
)

# 构建 FastAPI 应用，提供服务
app = FastAPI()

# 可选，前端报CORS时
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# 定义请求模型
class QuestionRequest(BaseModel):
    question: str


# 定义响应模型
class AnswerResponse(BaseModel):
    answer: str


# 提供查询接口 http://127.0.0.1:8000/ask
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # 获取用户问题
        user_question = request.question
        print(user_question)

        # 通过RAG链生成回答
        answer = qa_chain.run(user_question)

        # 返回答案
        answer = AnswerResponse(answer=answer)
        print(answer)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
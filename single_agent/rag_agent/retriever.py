# 문서 검색과 답변을 위한 도구 정의하기
# -> 랭그래프를 이용한 RAG 만들기

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings # 임베딩 모델
from langchain_chroma import Chroma # 벡터 스토어

DB_PATH = './chroma_db' # 벡터 스토어 경로

#벡터스토어 객체 생성
vectorstore = Chroma(
    # 사용자 프롬프트로 유사도 검색을 하려면, 사용자 프롬프트를 동일한 모델로 임베딩.
    embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'),
    # 벡터 DB 경로
    persist_directory=DB_PATH,
    # 유사도 검색할 대상 -> 지정을 하지 않으면 전체 검색을 해서, 검색 성능이 떨어짐
    collection_name='korean_pdf' # 문서가 여러개일때는 해당 name 관련된 것만 보는 것이 가능
)

vectorstore.get() # 벡터데이터베이스가 정상적으로 로드되는지 확인

retriever = vectorstore.as_retriever(search_kwargs={'k':3}) # 유사도 검색시 가장 유사한 문서 3개 반환

# 유사도 검색을 에이전트가 사용할 수 있는 도구 생성
from langchain_core.tools import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="pdf_search",
    # 도구 설명 -> llm이 도구 사용을 할 경우 설명 -> 한글 맞춤법 관련 질문이면 이 도구를 호출
    description='use this tool to search information from the Korean Spelling Rules PDF document',
)

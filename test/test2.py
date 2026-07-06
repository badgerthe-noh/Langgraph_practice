from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(model = 'gpt-4o-mini')


messages = [
    (
        "system", "당신은 사용자가 한 말을 영어로 번역하는 유능한 번역가 입니다."
    ),
    (
        "human", "안녕하세요"
    )
]

ai_message = llm.invoke(messages)
print(ai_message)
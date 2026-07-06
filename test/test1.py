# OpenAI 사용
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model = 'gpt-4o-mini',
    input = '대한민국의 수도는?'
)

print(response.output_text)


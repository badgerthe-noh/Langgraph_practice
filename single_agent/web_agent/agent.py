from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# 1. 필요한 도구와 도구 바인딩 모델을 생성
tool = TavilySearch(max_result = 3)
tools = [tool]

llm = ChatOpenAI(model = 'gpt-4o-mini')
llm_with_tools = llm.bind_tools(tools)

# 2. 상태 그래프 생성

# 그래프의 상태를 먼저 정의
# 간단히 메시지만 주고받으며 관리할 수 있도록 "messages"키를 가지는 데이터 스키마를 만듦.
# 이때 노드가 새로운 메시지를 업데이트 할 때마다 누적될 수 있도록 add_messages 리듀서 함수를 사용.
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 상태 그래프를 생성
graph_builder = StateGraph(State)

# 3. LLM 노드 생성
def chatbot(state: State):
    # 인보크 되는 것은 현재 상태의 messages 값 전체
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}

graph_builder.add_node('chatbot', chatbot)

# 4. 도구 노드 생성
from langchain.messages import ToolMessage
import json 

class BasicToolNode: 
    """
        마지막 AIMessage에서 요청된 도구를 실행하는 노드
    """
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs: dict):
        # chatbot 노드에서 업데이트 된 도구 호출 결과를 확인하기 위해 마지막 메시지를 불러옴
        # 이 메시지는 AIMessage로, tool_calls 속성을 가짐.
        # 해당 속성의 값을 도구의 입력으로 사용해 도구를 실행할 수 있음.
        if messages := inputs.get("messages", []): 
            message = messages[-1] # 마지막 메시지, AIMessage. 도구 호출 관련 내용
        else: 
            raise ValueError('ERROR: 입력된 메시지가 없습니다.')
        
        outputs = [] # 업데이트 할 메시지를 저장할 변수

        # 반복문을 사용하는 이유 -> messages.tool_calls 가 리스트이니까
        for tool_call in message.tool_calls:
            # 1. 도구 호출
            # 마지막 메시지의 tool_calls 속성을 확인하고, 호출된 도구를 불러와 도구 호출에 반환된 인자를 입력해 실행(invoke).
            # 도구 실행 결과는 tool_result에 저장되고, 이를 도구 메시지로 변환해 메시지 업데이트를 준비
            tool_result = self.tools_by_name[tool_call['name']].invoke(tool_call['args'])
            
            # 2. 도구 호출의 결과를 저장
            outputs.append(
                # 도구 메시지는 랭체인의 ToolMessage를 사용해 생성
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False), # 도구 호출의 결과
                    name = tool_call['name'], # 어떤 도구인지
                    tool_call_id = tool_call['id'], 
                )
            )
            # State가 업데이트 -> return은 ToolMessage만
        return {'messages': outputs}

tool_node = BasicToolNode(tools=[tool]) # 객체 생성
graph_builder.add_node('tools', tool_node)

# 5. 조건부 엣지 생성 -> 라우팅 함수를 만듦
# 1) 라우팅 함수
from langgraph.graph import START, END
def route_tools(state: State) -> str:
    """
        마지막 메시지에 도구 호출이 있는 경우, ToolNode로 라우팅하고 그렇지 않으면 END로 라우팅
    """

    # 1) AIMessage 추출
    if isinstance(state, list): # 매개변수로 받은 state가 리스트 형태이면
        ai_message = state[-1] # 최신 메시지
    elif messages := state.get('messages', []): # 매개변수로 받은 state가 딕셔너리라면 messages 키의 값을 추출
        ai_message = messages[-1]
    else:
        raise ValueError('f"ERROR: 입력에 메시지가 없습니다. 상태: {state}"')
    
    # 2) AIMessage에 tool_calls 있는지 확인
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return 'tools'
    return END

# 2) 조건부 엣지 생성
graph_builder.add_conditional_edges(
    'chatbot', # 시작 노드
    route_tools, # 라우팅 함수
    path_map= {'tools': 'tools', END: END} # key는 라우팅 함수의 리턴값, value는 이동할 노드
)

# 6. 엣지 생성 및 그래프 컴파일 
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('tools', 'chatbot')

graph = graph_builder.compile()

from pprint import pprint

if __name__ == '__main__':
    # 그래프 시각화 -> 별도의 파일로 저장 되도록
    try:
        image = graph.get_graph().draw_mermaid_png()
        with open('./graph.png', 'wb') as f:
            f.write(image)
    except Exception:
        pass

    # 결과
    response = graph.invoke({
        'messages': ['Langgraph가 무엇인가요?']
    })

    pprint(response)


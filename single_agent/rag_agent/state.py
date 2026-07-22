# 상태 정의 
# 에이전트 그래프에 필요한 상태를 정의
# 사용자의 질문을 다시 작성하기도 하고 검색된 문서와 답변의 적합성을 평가하기도 할 것.
# 그때마다 질문, 검색 컨택스트, 답변을 상태로 저장하고 불러오며 새로운 값으로 업데이트 하기 용이하도록 관리.

# 또 문서 관련성 평가에 통과하지 못해, 문서를 반복적으로 다시 검색하는 무한 루프에 갇히는 것을 방지.
# 이를 위해 retry_num 상태를 통해 반복횟수를 세고, 일정 횟수를 초과하면 문서 재탐색을 포기하도록 구현

from langgraph.graph import MessagesState

class AgentState(MessagesState): 
    question: str, # 질문
    context: str, # 검색 컨텍스트
    answer: str, # 답변
    retry_num: int # 검색 횟수 -> 무한루프 방지. 일정 횟수 이상은 검색 안함
from client import llm
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
runnable_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

reponse=runnable_with_history.invoke(
    [HumanMessage(content="hi - im bob!")],
    config={"configurable": {"session_id": "1"}},
)
print(reponse.content)

reponse_session1=runnable_with_history.invoke(
    [HumanMessage(content="whats my name?")],
    config={"configurable": {"session_id": "1"}},
)
print(f"session1-: {reponse_session1.content}")

reponse_session2=runnable_with_history.invoke(
    [HumanMessage(content="whats my name?")],
    config={"configurable": {"session_id": "2"}},
)
print(f"session1-: {reponse_session2.content}")




from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from client import llm


joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | llm
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | llm
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

response=map_chain.invoke({"topic": "bear"})

print(response)
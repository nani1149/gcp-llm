from client import llm
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "funny"}))
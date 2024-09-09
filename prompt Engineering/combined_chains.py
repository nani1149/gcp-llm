from client import llm
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | llm | StrOutputParser()

analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

composed_chain = {"joke": chain} | analysis_prompt | llm | StrOutputParser()

reponse=composed_chain.invoke({"topic": "bears"})
print(reponse)

from client import llm
from pdf_retriver import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """

You are a information retrieval AI. Format the retrieved information as a table or text


Use only the context for your answers, do not make up information

query: {query}

{context} 
"""
# Converts the prompt into a prompt template
prompt = ChatPromptTemplate.from_template(template)
#Using OpenAI model, by default gpt 3.5 Turbo
model = llm
chain = (
# The initial dictionary uses the retriever and user supplied query
    {"context":retriever,
     "query":RunnablePassthrough()}
# Feeds that context and query into the prompt then model & lastly 
# uses the ouput parser, do query for the data.
    |  prompt  | model | StrOutputParser()
)

reponse=chain.invoke("""A differential form on a complex orbifold Z is 
""")
print(reponse)
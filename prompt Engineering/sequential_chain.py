from client import llm
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# decouple to read .env variables(OpenAI Key)
# simple sequential chain
from langchain.chains import SimpleSequentialChain
# memory in sequential chain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from datetime import datetime



# rapper
rapper_template: str = """You are an American rapper, your job is to come up with\
lyrics based on a given topic

Here is the topic you have been asked to generate a lyrics on:
{input}\
"""

rapper_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["input"], template=rapper_template)

# creating the rapper chain
rapper_chain: LLMChain = LLMChain(
    llm=llm, output_key="lyric", prompt=rapper_prompt_template)

# verifier
verifier_template: str = """You are a verifier of rap songs, you are tasked\
to inspect the lyrics of rap songs. If they consist of violence and abusive languge\
flag the lyrics. 

Your response should be in the following format, as a Python Dictionary.
lyric: this should be the lyric you received 
Violence_words: True or False

Here is the lyrics submitted to you:
{lyric}\
"""

verified_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["lyric"], template=verifier_template)

# creating the verifier chain
verifier_chain: LLMChain = LLMChain(
    llm=llm, output_key="AI_verified", prompt=verified_prompt_template)


# final output chain
final_template: str = """You are a final quality assurance of a lyrics post.\
Your job will be to accept a lyric and output data in the following format

Your final response should be in the following format, in a Python Dictionary format:
lyric: this should be the lyric you received
Date and time verified: {time_created_and_verified}
Verified by human: {verified_by_human}

Here is the lyric submitted to you:
{AI_verified}\
"""

final_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["AI_verified", "time_created_and_verified", "verified_by_human"], template=final_template)

# creating the verifier chain
final_chain: LLMChain = LLMChain(
    llm=llm, output_key="final_output", prompt=final_prompt_template)


# creating the simple sequential chain
ss_chain: SequentialChain = SequentialChain(
    memory=SimpleMemory(memories={
                        "time_created_and_verified": str(datetime.utcnow()), "verified_by_human": "False"}),
    chains=[rapper_chain, verifier_chain, final_chain],
    # multiple variables
    input_variables=["input"],
    output_variables=["final_output"],
    verbose=True)

# running chain
review = ss_chain.run(input="christ worship songs")
print(review)
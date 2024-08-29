import openai
from google.auth import default, transport
from langchain import PromptTemplate
# Build
from langchain_openai import ChatOpenAI
from vertexai.preview import rag

credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)


MODEL_LOCATION = "us-central1"
PROJECT_ID='sacred-alliance-433217-e3'
MODEL_ID = "meta/llama3-405b-instruct-maas"  # @param {type:"string"} ["meta/llama3-405b-instruct-maas"]

client = openai.OpenAI(
    base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    api_key=credentials.token,
)
# response = client.chat.completions.create(
#     model=MODEL_ID, messages=[{"role": "user", "content": "Complete the sentence: The sky like "}]
# )
#Information Extract
prompt=f"Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.Mention the large language model based product mentioned in the paragraph above:"
#Question and answer 
qa= f"Answer the question based on the context below. Keep the answer short and concise. Respond 'Unsure about answer' if not sure about the answer.Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use. Question: What was OKT3 originally sourced from? Answer:"
# Text Classification
classification="Classify the text into neutral, negative or positive. Text: I think the food was okay. Sentiment:"
response = client.chat.completions.create(
    model=MODEL_ID, messages=[{"role": "user", "content": {classification}}]
)
print(response.choices[0].message.content)
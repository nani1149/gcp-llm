from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from client import llm
from langchain_chroma import Chroma

# Load the PDF
loader = PyPDFLoader("sample_pdf.pdf")
pages = loader.load_and_split()  # Load and split the document into pages

# Extract only the text content from the pages (if pages is a list of objects, we extract page_content)
page_texts = [page.page_content for page in pages]  # Extract the text content of each page

# Print the first page's content to check
#print(page_texts[0])

# Now split the text using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Maximum size of each chunk
    chunk_overlap=10,  # Overlap between chunks
    length_function=len,  # Function to calculate length
    separators=["\n\n"]  # Separators to use for splitting
)

# Split the text content into chunks
docs = text_splitter.create_documents(page_texts)  # Split based on the extracted text
#print(texts[0].page_content)  # Print the first chunk

#text = "LangChain is the framework for building context-aware reasoning applications"

# Initialize the a specific Embeddings Model version
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

db = Chroma.from_documents(docs, embeddings)
query = "A differential form on a complex orbifold Z is "
answer = db.similarity_search(query)
#print(answer)

# Building the retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# single_vector = embeddings.embed_query(texts[0].page_content)
# print(str(single_vector)[:100])
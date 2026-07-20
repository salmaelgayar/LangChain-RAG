from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

loader = DirectoryLoader("data")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="chroma_db"
)

db.persist()

print("Vector database created.")

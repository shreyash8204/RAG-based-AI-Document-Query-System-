import os

import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


# -----------------------------------------
# 1. Load environment variables
# -----------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# -----------------------------------------
# 2. Configure Gemini
# -----------------------------------------

# Gemini LLM
llm = GoogleGenAI(
    model="gemini-3.6-flash",
    api_key=API_KEY,
)

# Gemini Embedding Model
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
    api_key=API_KEY,
)

# Set LlamaIndex models
Settings.llm = llm
Settings.embed_model = embed_model


# -----------------------------------------
# 3. Data directory
# -----------------------------------------

DATA_DIR = "./my_data"

os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------------------
# 4. Create sample document
# -----------------------------------------

sample_text = """
Core2web Technologies is a premier coding academy
and IT training institute founded in January 2017
by Shashi Bagal Sir.

The headquarters is located in Pune, Maharashtra.

The institute focuses on technical logic building,
coding fundamentals, and industry-oriented software
engineering.

Core offerings include C, C++, Java, Python,
Data Structures and Algorithms, Flutter, React,
Spring Boot, DBMS and Operating Systems.

The institute also provides placement assistance,
mock interviews and aptitude preparation.
"""

file_path = os.path.join(DATA_DIR, "academy_info.txt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(sample_text.strip())


# -----------------------------------------
# 5. Connect to ChromaDB
# -----------------------------------------

db_client = chromadb.PersistentClient(
    path="./chroma_db"
)

chroma_collection = db_client.get_or_create_collection(
    "academy_knowledge_base"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# -----------------------------------------
# 6. Load documents
# -----------------------------------------

documents = SimpleDirectoryReader(
    DATA_DIR
).load_data()


# -----------------------------------------
# 7. Create vector index
# -----------------------------------------

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)


# -----------------------------------------
# 8. Create query engine
# -----------------------------------------

query_engine = index.as_query_engine(
    similarity_top_k=5
)


# -----------------------------------------
# 9. Ask question
# -----------------------------------------

# question = "Who founded Core2web Technologies and where is it located?"

# print(f"\nQuestion: {question}")

# response = query_engine.query(question)

# print(f"\nAnswer:\n{response}")

print("\nRAG System Ready!")
print("Ask questions about Core2web Technologies.")
print("Type 'exit' to stop.\n")

while True:

    question = input("Question: ")

    if question.lower() == "exit":
        print("RAG system stopped.")
        break

    response = query_engine.query(question)

    print(f"\nAnswer: {response}\n")

# RAG System Ready!
# Ask questions about Core2web Technologies.
# Type 'exit' to stop.

# Question: Who founded Core2web Technologies?

# Answer: Core2web Technologies was founded by Shashi Bagal Sir.

# Question: When was it founded?

# Answer: It was founded in January 2017.

# Question: Where is it located?

# Answer: Its headquarters is located in Pune, Maharashtra.

# Question: What courses are available?

# Answer: The institute offers C, C++, Java, Python,
# Data Structures and Algorithms, Flutter, React,
# Spring Boot, DBMS and Operating Systems.

# Question: exit

# RAG system stopped.
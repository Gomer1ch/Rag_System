import os
from dotenv import load_dotenv

load_dotenv()  # читаємо .env

#OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
#UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME    = "documents3"
EMBEDDING_MODEL    = "intfloat/multilingual-e5-base"
#EMBEDDING_MODEL    = "text-embedding-ada-002"

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 100
TOP_K              = 5


#print("OPENAI:", OPENAI_API_KEY)
print("QDRANT:", QDRANT_URL)
import os
from dotenv import load_dotenv
load_dotenv()
 
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")  # or "hf"
 
# For OpenAI embeddings and LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
 
# HuggingFace model name for local embeddings (if EMBEDDING_PROVIDER=hf)
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
 
# Directory containing plain text files (or pdfs if you extend loaders)
DATA_DIR = os.environ.get("DATA_DIR", "./data")
 
# Where to persist FAISS index
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "./faiss_index")
 
# retriever settings
TOP_K = int(os.environ.get("TOP_K", "4"))
 
# Chat LLM settings
USE_CHAT_OPENAI = os.environ.get("USE_CHAT_OPENAI", "true").lower() in ("1","true","yes")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
import os
import pickle
from typing import Tuple
 
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
 
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
 
import config
 
print()
 
def _get_embeddings():
   
    if config.EMBEDDING_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in environment for openai embeddings.")
        return OpenAIEmbeddings()
    else:
        # HuggingFace local embedding model (sentence-transformers)
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
 
 
def ingest_from_data_dir(data_dir: str = None, persist_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    data_dir = data_dir or config.DATA_DIR
    persist_dir = persist_dir or config.FAISS_INDEX_DIR
    # Load text files (simple). You can add PDF/Markdown loaders.
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
 
    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)
 
    embeddings = _get_embeddings()
    faiss_index = FAISS.from_documents(docs, embeddings)
 
    # persist
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
    faiss_index.save_local(persist_dir)
    # Save metadata docs if you want (optional)
    print(f"Saved FAISS index to: {persist_dir}")
    return faiss_index
 
 
def load_faiss(persist_dir: str = None) -> FAISS:
    persist_dir = persist_dir or config.FAISS_INDEX_DIR
    embeddings = _get_embeddings()
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"{persist_dir} does not exist. Run ingest first.")
    faiss_index = FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)
    return faiss_index
 
 
def get_retriever(persist_dir: str = None, search_kwargs: dict = None):
    faiss = load_faiss(persist_dir)
    retriever = faiss.as_retriever(search_kwargs=search_kwargs or {"k": config.TOP_K})
    return retriever
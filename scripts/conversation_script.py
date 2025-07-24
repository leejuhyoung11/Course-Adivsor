# scripts/conversation_script.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.mistral_loader import load_mistral_pipeline  
from utils.hf_auth import hf_login_from_env


def load_vectorstore(persist_dir: str, embedding_model: str):
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


def build_prompt(context: str, query: str) -> str:
    return f"""You are a helpful assistant for answering questions from a university course catalog.

Context:
{context}

Question: {query}
Answer:"""


def main():
    
    hf_login_from_env()
    
    persist_dir = "vectorstore/catalog_chroma_db"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading vector store...")
    vectorstore = load_vectorstore(persist_dir, embedding_model)
    retriever = vectorstore.as_retriever(search_type="mmr")


if __name__ == "__main__":
    main()
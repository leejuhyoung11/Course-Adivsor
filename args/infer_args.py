# args/infer_args.py

import argparse

def get_infer_args():
    parser = argparse.ArgumentParser(description="Run RAG inference with Mistral and Chroma")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--persist_path", type=str, default="vectorstore/catalog_chroma_db", help="Chroma vector DB path")
    return parser
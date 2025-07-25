# scripts/rag_infer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.mistral_loader import load_mistral_pipeline  
from utils.hf_auth import hf_login_from_env
from args.vector_args import get_path_args

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
    parser = get_path_args()
    args = parser.parse_args()
    
    pdf_vector_path = args.pdf_vector_path
    
    embedding_model = args.embedder_name
    mistral_model = args.model_name

    print("Loading vector store...")
    vectorstore = load_vectorstore(pdf_vector_path, embedding_model)

    print("Loading Mistral model...")
    llm_pipe = load_mistral_pipeline(mistral_model)

    query = input("\nEnter your question: ")

    docs = vectorstore.similarity_search(query, k=10)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt(context, query)

    print("\n Prompt:\n", prompt)
    result = llm_pipe(prompt)[0]["generated_text"]

    print("\n Mistral's Answer:")
    print(result.split("Answer:")[-1].strip(), flush=True)
    # print(result.strip(), flush=True)


if __name__ == "__main__":
    main()
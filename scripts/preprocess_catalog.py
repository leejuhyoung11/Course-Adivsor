# scripts/preprocess_catalog.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from args.vector_args import get_vectorstore_args
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    parser = get_vectorstore_args()
    args = parser.parse_args()

    print(f"Loading PDF from {args.pdf_path}")
    loader = PyPDFLoader(args.pdf_path)
    documents = loader.load()

    print(f"Splitting into chunks (size={args.chunk_size}, overlap={args.chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    print(f"Embedding with model: {args.model_name}")
    embedder = HuggingFaceEmbeddings(model_name=args.model_name)
    
    
    print(f"Saving vectorstore to {args.save_path}")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedder,
        persist_directory=args.save_path
    )
    vectorstore.persist()

    
    
    print("Done")


if __name__ == "__main__":
    main()
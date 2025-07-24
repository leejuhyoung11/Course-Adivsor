# scripts/preprocess_catalog.py

import sys
import os, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from args.vector_args import get_vectorstore_args
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def parse_courses_by_hours_line(text) -> list[Document]:
    lines = text.splitlines()
    courses = []
    current_title = None
    current_desc = []

    pattern = re.compile(r"\((\d+(?:-\d+)? Hours)\)")

    for line in lines:
        line = line.strip().replace("\xa0", " ")
        if not line:
            continue

        if pattern.search(line): 
            # Saving Previous course info
            if current_title:
                full_desc = " ".join(current_desc).strip()
                full_text = f"{current_title}\n{full_desc}".strip()
                prereq_match = re.search(r"Prerequisite\(s\): (.+?)(?:\.|\Z)", full_desc)
                prereq = prereq_match.group(1).strip() if prereq_match else None

                courses.append(Document(
                    page_content=full_text,
                    metadata={
                        "title_line": current_title,
                        "prerequisite": prereq
                    }
                ))

            current_title = line
            current_desc = []

        else:
            current_desc.append(line)

    # Adding Last Element in file
    if current_title:
        full_desc = " ".join(current_desc).strip()
        full_text = f"{current_title}\n{full_desc}".strip()
        prereq_match = re.search(r"Prerequisite\(s\): (.+?)(?:\.|\Z)", full_desc)
        prereq = prereq_match.group(1).strip() if prereq_match else None

        courses.append(Document(
            page_content=full_text,
            metadata={
                "title_line": current_title,
                "prerequisite": prereq
            }
        ))

    return courses

def main():
    parser = get_vectorstore_args()
    args = parser.parse_args()

    print(f"Loading PDF from {args.pdf_path}")
    loader = PyPDFLoader(args.pdf_path)
    documents = loader.load()

    print(f"Embedding with model: {args.model_name}")
    embedder = HuggingFaceEmbeddings(model_name=args.model_name)
    
    document_list = []
    for d in documents:
        arr = parse_courses_by_hours_line(d.page_content)
        document_list.extend(arr)
    
    print(f"Saving vectorstore to {args.save_path}")
    vectorstore = Chroma.from_documents(
        documents=document_list,
        embedding=embedder,
        persist_directory=args.save_path
    )   
    print("Done")


if __name__ == "__main__":
    main()
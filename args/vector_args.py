# args/vector_args.py

import argparse

def get_vectorstore_args():
    parser = argparse.ArgumentParser(description="PDF to Chroma Vectorstore")
    parser.add_argument("--pdf_path", type=str, help="Path to input PDF", default='data/raw/course_catalog_2023_2024.pdf')
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model name")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for splitting")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap for splitting")
    parser.add_argument("--save_path", type=str, default="vectorstore/catalog", help="Directory to save Chroma index")
    return parser

def get_course_schedule_args():
    parser = argparse.ArgumentParser(description="JSON file to Chroma Vectorstore")
    parser.add_argument("--json_path", type=str, help="Path to input PDF", default='data/raw/course_schedule.json')
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model name")
    parser.add_argument("--save_path", type=str, default="vectorstore/course_schedule", help="Directory to save Chroma index")
    return parser
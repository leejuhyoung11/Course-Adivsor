
import sys
import os, re, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from args.vector_args import get_course_schedule_args
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def format_course_text(course):
    return f"""{course["title"]}
Subject: {course["subject"]}
Course Number: {course["course_number"]} Section: {course["section"]} ({course["credit_hours"]} credits)
CRN: {course["crn"]}
Term: {course["term"]}
Meeting Time: {course["meeting_time"]}
Campus: {course["campus"]}
Status: {course["status"]}
Attributes: {course["attribute"]}
""".strip()

def main():
    parser = get_course_schedule_args()
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        course_data = json.load(f)

    docs = [
    Document(
        page_content=format_course_text(course),
        metadata={"course_code": course["course_number"]}
        )
        for course in course_data
    ]

    embedder = HuggingFaceEmbeddings(model_name=args.model_name)
    db = Chroma.from_documents(docs, embedding=embedder, persist_directory=args.save_path)

    print("Done")


if __name__ == "__main__":
    main()
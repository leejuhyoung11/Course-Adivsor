# scripts/conversation_script.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from models.mistral_loader import load_mistral_pipeline  
from utils.hf_auth import hf_login_from_env
from args.vector_args import get_path_args

from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap


def load_vectorstore(persist_dir: str, embedding_model: str):
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)

def load_jsonstore(persist_dir: str, embedding_model: str):
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
    json_vector_path = args.json_vector_path
    embedding_model = args.embedder_name
    mistral_model = args.model_name

    print("Loading Mistral model...")
    llm_pipe = load_mistral_pipeline(mistral_model)

    print("Loading vector store...")
    vectorstore = load_vectorstore(pdf_vector_path, embedding_model)
    retriever1 = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 7})
    jsonstore = load_jsonstore(json_vector_path, embedding_model)
    


    def first_step(query):
        docs = retriever1.invoke(query)
        allowed_courses = [doc.metadata["course_title"] for doc in docs]
        return {"docs": docs, "allowed_courses": allowed_courses, "query":query}

    def second_step(data):
        query = data["query"]
        allowed_courses = data["allowed_courses"]
        title_filter = {"course_title": {"$in": allowed_courses}}
        retriever2 = jsonstore.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'filter': title_filter})
        return retriever2.invoke(query)

    pipeline = RunnableSequence(
        RunnableLambda(first_step),
        RunnableLambda(second_step)
    )

    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an academic assistant. Answer based on the following course catalog and schedule context.\n"
         "Relevant Class schedule Info:\n{schedule_info}"),
        MessagesPlaceholder("chat_history"),
        ("human", "My question is: {question}\nPlease answer only based on context.")
    ])
    
    llm_chain = prompt | llm_pipe 
    

    chat_history = []

    while True:
        user_input = input("\n💬 User: ")
    
        if user_input.lower() in ["exit", "quit"]:
            print("👋 대화를 종료합니다.")
            break

        schedule_docs = pipeline.invoke(user_input)

        docs_formatted = {
            "schedule_info": "\n".join([doc.page_content for doc in schedule_docs])
        }
    
      
        prompt_text = prompt.format(
        chat_history=chat_history,
        question=user_input,
        **docs_formatted
        )
        print("✅ LLM response received!")
    
        # LLM pipeline 실행
        response_text = llm_pipe(prompt_text)[0]["generated_text"]
        print("\n🤖 LLM:", response_text)
    
        # 히스토리 업데이트
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response_text)
        ])


if __name__ == "__main__":
    main()
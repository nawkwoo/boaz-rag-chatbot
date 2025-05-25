from query.query_pipeline import get_top_documents
from langchain.prompts import ChatPromptTemplate
from config import RERANKING_STRATEGY, TOP_K

from langchain.llms import Ollama

def get_llm():
    return Ollama(model="llama3", temperature=0.2)  # ollama 로컬 모델 이름

def get_final_answer(query: str) -> str:
    docs = get_top_documents(query)
    context = "\n\n".join(docs)

    prompt = ChatPromptTemplate.from_template(
        """
        아래 문서를 참고하여 사용자의 질문에 답변하세요.

        [문서]
        {context}

        [질문]
        {question}

        [답변]
        """
    )

    chain = prompt | get_llm()
    return chain.invoke({"context": context, "question": query})

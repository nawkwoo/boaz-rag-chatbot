import os
from dotenv import load_dotenv
import google.generativeai as genai
from query.query_pipeline import get_top_documents
from config import TOP_K_FINAL
from langchain.schema import Document

load_dotenv()

# Gemini API 키 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 모델 선택 (혹시나)
model = genai.GenerativeModel("gemini-2.0-flash")
# model = genai.GenerativeModel("gemini-1.5-flash")
# model = genai.GenerativeModel("gemini-pro")

def get_final_answer(query: str) -> str:
    docs: list[Document] = get_top_documents(query)
    
    print(f"🔍 관련 문서 개수: {len(docs)}")
    print(f"📝 프롬프트 context 예시:\n{docs[0].page_content[:300] if docs else '문서 없음'}")

    # ✅ Document.page_content로 context 구성
    context = "\n\n".join([doc.page_content for doc in docs])[:1000]

    prompt = f"""
    아래 문서를 참고하여 사용자의 질문에 답변하세요.

    [문서]
    {context}

    [질문]
    {query}

    [답변]
    """

    response = model.generate_content(prompt)
    return response.text.strip()
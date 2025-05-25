from transformers import pipeline
from query.query_pipeline import get_top_documents
from config import RERANKING_STRATEGY, TOP_K

llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def get_final_answer(query: str) -> str:
    docs = get_top_documents(query)  # str 리스트
    context = "\n\n".join(docs)[:1000]

    prompt = f"""
    아래 문서를 참고하여 사용자의 질문에 답변하세요.

    [문서]
    {context}

    [질문]
    {query}

    [답변]
    """

    result = llm_pipeline(prompt, max_new_tokens=300, do_sample=False)
    print("🔍 Raw result:", result)
    return result[0]["generated_text"]

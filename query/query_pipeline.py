# query_pipeline.py

from config import TOP_K_FINAL, USE_RERANKING
from vectorization.search import search_documents
from query.reranker import rerank

def get_top_documents(query: str) -> list[str]:
    """검색 후 config 설정에 따라 재정렬 결과 반환"""
    docs = search_documents(query, top_k=TOP_K_FINAL * 2)  # 넉넉히 후보 확보
    texts = [doc["text"] for doc in docs]

    if USE_RERANKING:
        reranked = rerank(query, texts, top_k=TOP_K_FINAL)
        return [text for text, _ in reranked]
    else:
        return texts[:TOP_K_FINAL]
from vectorization.search import search_documents
from query.reranker import rerank
from config import USE_RERANKING, TOP_K

def get_top_documents(query: str) -> list[str]:
    """질문에 대한 관련 문서 추출 (검색 + 옵션 재정렬)"""
    docs = search_documents(query, top_k=TOP_K * 2)
    texts = [doc["text"] for doc in docs]

    if USE_RERANKING:
        reranked = rerank(query, texts, top_k=TOP_K)
        return [doc for doc, _ in reranked]
    else:
        return texts[:TOP_K]

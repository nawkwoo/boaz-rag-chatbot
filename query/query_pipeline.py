# query_pipeline.py

from config import TOP_K_FINAL, USE_RERANKING
from vectorization.search import search_documents
from query.reranker import rerank
from langchain.schema import Document

def get_top_documents(query: str) -> list[Document]:
    """검색 후 config 설정에 따라 재정렬 결과 반환"""
    docs = search_documents(query, top_k=TOP_K_FINAL * 2)

    if USE_RERANKING:
        # 🔁 rerank는 텍스트 리스트 기반
        texts = [doc.page_content for doc in docs]
        reranked = rerank(query, texts, top_k=TOP_K_FINAL)

        # 🔁 다시 Document 형태로 매핑
        reranked_docs = []
        text_to_doc = {doc.page_content: doc for doc in docs}
        for text, _ in reranked:
            doc = text_to_doc.get(text, Document(page_content=text, metadata={}))
            reranked_docs.append(doc)

        return reranked_docs
    else:
        return docs[:TOP_K_FINAL]

import sys
import os

# 상위 디렉토리에서 config, retriever 모듈들을 import할 수 있도록 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    USE_SPARSE,          # True면 Sparse (BM25), False면 Dense (SBERT) 사용
    TOP_K,               # 검색 결과 상위 K개 문서 반환
    DENSE_INDEX_NAME,
    SPARSE_INDEX_NAME,
)

from retriever.dense_retriever import DensePineconeRetriever
from retriever.sparse_retriever import create_sparse_retriever

def create_retriever():
    """
    설정에 따라 Dense 또는 Sparse Retriever 인스턴스를 생성하여 반환합니다.
    - config.USE_SPARSE = True → Sparse (BM25 기반)
    - config.USE_SPARSE = False → Dense (SBERT 기반)
    """
    if USE_SPARSE:
        print("🔍 Sparse Retriever 사용 중 (BM25)")
        return create_sparse_retriever()
    else:
        print("🔍 Dense Retriever 사용 중 (SBERT)")
        return DensePineconeRetriever(index_name=DENSE_INDEX_NAME, top_k=TOP_K)

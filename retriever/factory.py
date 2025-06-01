import sys
import os

# 상위 디렉토리에서 config 등을 import할 수 있도록 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    USE_SPARSE,
    TOP_K,
    DENSE_INDEX_NAME,
    SPARSE_INDEX_NAME,
)

from retriever.dense_retriever import DensePineconeRetriever
from retriever.sparse_retriever import create_sparse_retriever

def create_retriever():
    if USE_SPARSE:
        print("🔍 Sparse Retriever 사용 중 (BM25)")
        return create_sparse_retriever()
    else:
        print("🔍 Dense Retriever 사용 중 (SBERT)")
        return DensePineconeRetriever(index_name=DENSE_INDEX_NAME, top_k=TOP_K)

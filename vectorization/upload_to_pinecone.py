# upload_to_pinecone.py

from config import INDEXING_STRATEGY
from vectorization.dense_index import upload_dense_index
from vectorization.sparse_index import upload_sparse_index

if __name__ == "__main__":
    if INDEXING_STRATEGY == "dense":
        upload_dense_index()
    elif INDEXING_STRATEGY == "sparse":
        upload_sparse_index()
    elif INDEXING_STRATEGY == "hybrid":
        upload_dense_index()
        upload_sparse_index()
    else:
        raise ValueError(f"[❌ 에러] 지원되지 않는 인덱싱 전략: {INDEXING_STRATEGY}")
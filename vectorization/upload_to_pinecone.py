### upload_to_pinecone.py

from config import INDEXING_STRATEGY
from vectorization.dense_index import upload_dense_index
from vectorization.sparse_index import upload_sparse_index

def run_index_upload():
    print("🚀 [2/2] Pinecone 벡터 업로드 시작")
    if INDEXING_STRATEGY == "dense":
        upload_dense_index()
    elif INDEXING_STRATEGY == "sparse":
        upload_sparse_index()
    elif INDEXING_STRATEGY == "hybrid":
        upload_dense_index()
        upload_sparse_index()
    else:
        raise ValueError(f"❌ 지원되지 않는 INDEXING_STRATEGY: {INDEXING_STRATEGY}")

if __name__ == "__main__":
    run_index_upload()

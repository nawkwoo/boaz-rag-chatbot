from preprocess.builder import process_all_documents
from vectorization.upload_to_pinecone import upload_dense_index, upload_sparse_index
from config import INDEXING_STRATEGY

# 1. 전처리 실행 (data → data_with_meta)
print("🚀 [1/2] 전처리 시작")
process_all_documents("data", "data_with_meta")

# 2. 벡터 업로드
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

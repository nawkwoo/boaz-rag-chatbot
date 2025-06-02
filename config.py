# Sparse (BM25) 사용 여부
USE_SPARSE = True  # False면 Dense 사용

# Dense 설정
# DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
DENSE_MODEL_NAME = "jhgan/ko-sbert-sts"
DENSE_INDEX_NAME = "boaz-dense-index"
ID_TO_TEXT_PATH_DENSE = "data_with_meta/id_to_text_dense.json"

# Sparse 설정
SPARSE_MODEL_NAME = "pinecone-sparse-english-v0"
SPARSE_INDEX_NAME = "boaz-bm25-index"
ID_TO_TEXT_PATH_SPARSE = "data_with_meta/id_to_text_sparse.json"

# 공통 설정
TOP_K = 20
DATA_PATH = "data"

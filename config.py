# config.py

# 인덱싱 전략: "dense", "sparse"
INDEXING_STRATEGY = "dense"

# 사용할 임베딩 모델 (SBERT 기반)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# SBERT 임베딩 차원
EMBEDDING_DIMENSION = 384

# Pinecone 관련 설정
INDEX_NAME = "boaz-index"
BM25_INDEX_NAME = "boaz-bm25-index"

# 재랭킹 설정
USE_RERANKING = True
RERANKING_STRATEGY = "cross-encoder"  # or "sbert"
TOP_K = 5

# 청크 관련
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10

# 저장된 전처리 결과 위치
DATA_PATH = "data_with_meta"
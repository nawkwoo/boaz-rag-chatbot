# config.py

# 인덱싱 전략: "dense", "sparse", "hybrid"
INDEXING_STRATEGY = "sparse"

# 사용할 임베딩 모델 (SBERT 기반)
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# 더 강한 모델 추천 (우선 유지해보자)
# cross-encoder/ms-marco-MiniLM-L-12-v2
# cross-encoder/stsb-roberta-base
# cross-encoder/qnli-roberta-base

# SBERT 임베딩 차원 (정확한 모델 차원으로 수정)
EMBEDDING_DIMENSION = 384

# Pinecone 관련 설정
DENSE_INDEX_NAME = "boaz-index"
SPARSE_INDEX_NAME = "boaz-bm25-index"

# 각 단계별 top-k 설정
TOP_K_BM25 = 7       # BM25 검색 시 top-k
TOP_K_SBERT = 5      # SBERT 재랭킹 시 top-k
TOP_K_FINAL = 3      # 최종 Cross-Encoder 선택 수

# 재랭킹 설정: "cross-encoder", "sbert" (이게 코사인)
USE_RERANKING = True
RERANKING_STRATEGY = "cross-encoder"

# 청크 관련
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10

# 저장된 전처리 결과 위치
DATA_PATH = "data_with_meta"
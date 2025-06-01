# config.py

# 임베딩 모델 이름 (HuggingFace SBERT)
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

# Pinecone 인덱스 이름
DENSE_INDEX_NAME = "boaz-dense-index"

# top-k 설정
TOP_K = 20

# 데이터 경로
DATA_PATH = "data"
ID_TO_TEXT_PATH = "data_with_meta/id_to_text_dense.json"
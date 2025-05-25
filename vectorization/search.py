import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from config import INDEXING_STRATEGY, INDEX_NAME, BM25_INDEX_NAME, EMBEDDING_MODEL

load_dotenv()

# 모델 초기화
sbert_model = SentenceTransformer(EMBEDDING_MODEL)
bm25_encoder = BM25Encoder()

# Pinecone 클라이언트 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 인덱스 연결
if INDEXING_STRATEGY == "dense":
    index = pc.Index(INDEX_NAME)
elif INDEXING_STRATEGY == "sparse":
    index = pc.Index(BM25_INDEX_NAME)
else:
    raise ValueError(f"[❌ 에러] 알 수 없는 전략: {INDEXING_STRATEGY}")

def search_documents(query: str, top_k: int = 10) -> list[dict]:
    """질문에 대해 dense 또는 sparse 방식으로 Pinecone에서 유사 문서 검색"""
    if INDEXING_STRATEGY == "dense":
        query_vec = sbert_model.encode([query])[0].tolist()
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

    elif INDEXING_STRATEGY == "sparse":
        query_sparse = bm25_encoder.encode_queries([query])[0]
        results = index.query(sparse_vector=query_sparse, top_k=top_k, include_metadata=True)

    else:
        raise ValueError(f"[❌ 에러] 지원되지 않는 검색 전략: {INDEXING_STRATEGY}")

    return [
        {
            "score": match["score"],
            "text": match["metadata"].get("text", ""),
            "metadata": match["metadata"]
        }
        for match in results["matches"]
    ]

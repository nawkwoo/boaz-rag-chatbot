import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document

from config import (
    DATA_PATH,
    SPARSE_INDEX_NAME,
    DENSE_INDEX_NAME,
    DENSE_MODEL_NAME,
    INDEXING_STRATEGY,
    TOP_K_FINAL
)
from vectorization.utils import truncate_sparse_vector

load_dotenv()

bm25_encoder = BM25Encoder()
_fit_done = False
sbert_model = SentenceTransformer(DENSE_MODEL_NAME)

# ID → 텍스트 매핑 파일 로드
strategy_to_path = {
    "dense": "id_to_text_dense.json",
    "sparse": "id_to_text_sparse.json",
    "hybrid": "id_to_text_dense.json"  # hybrid는 dense 기준
}
id_to_text_path = strategy_to_path.get(INDEXING_STRATEGY, "id_to_text_dense.json")

if not os.path.exists(id_to_text_path):
    raise FileNotFoundError(f"❌ '{id_to_text_path}' 파일이 없습니다. run.py 또는 업로드 스크립트를 먼저 실행하세요.")

with open(id_to_text_path, encoding="utf-8") as f:
    ID_TO_TEXT = json.load(f)

def load_texts_from_jsonl(folder_path: str):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".jsonl"):
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    texts.append(record["text"])
    return texts

def search_documents(query: str, top_k: int = TOP_K_FINAL * 2) -> list[Document]:
    if INDEXING_STRATEGY == "sparse":
        return search_sparse(query, top_k)
    elif INDEXING_STRATEGY == "dense":
        return search_dense(query, top_k)
    elif INDEXING_STRATEGY == "hybrid":
        return search_hybrid(query, top_k)
    else:
        raise ValueError(f"❌ 지원되지 않는 INDEXING_STRATEGY: {INDEXING_STRATEGY}")

def search_sparse(query: str, top_k: int) -> list[Document]:
    global _fit_done
    if not _fit_done:
        texts = load_texts_from_jsonl(DATA_PATH)
        bm25_encoder.fit(texts)
        _fit_done = True

    query_sparse = bm25_encoder.encode_queries([query])[0]
    query_sparse = truncate_sparse_vector(query_sparse)

    if len(query_sparse["indices"]) == 0:
        print("❌ Sparse 벡터가 비어 있어 검색 불가")
        return []

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(SPARSE_INDEX_NAME)

    results = index.query(
        vector=None,
        sparse_vector=query_sparse,
        top_k=top_k,
        include_metadata=True
    )

    return [
        Document(
            page_content=ID_TO_TEXT.get(match["id"], ""),
            metadata=match.get("metadata", {})
        )
        for match in results.get("matches", [])
    ]

def search_dense(query: str, top_k: int) -> list[Document]:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(DENSE_INDEX_NAME)

    query_vec = sbert_model.encode([query])[0].tolist()
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )

    matches = results.get("matches", [])
    for match in matches:
        print(f"{match['id']} | {match['score']:.4f} | {ID_TO_TEXT.get(match['id'], '')[:80]}")

    return [
        Document(
            page_content=ID_TO_TEXT.get(match["id"], ""),
            metadata=match.get("metadata", {})
        )
        for match in matches
    ]

def search_hybrid(query: str, top_k: int) -> list[Document]:
    sparse_results = search_sparse(query, top_k)
    dense_results = search_dense(query, top_k)

    # 중복 제거 (텍스트 기준)
    all_docs = {doc.page_content: doc for doc in sparse_results + dense_results}

    texts = list(all_docs.keys())
    doc_vecs = sbert_model.encode(texts)
    query_vec = sbert_model.encode([query])[0].tolist()

    scores = cosine_similarity([query_vec], doc_vecs)[0]
    sorted_indices = scores.argsort()[::-1][:TOP_K_FINAL]

    reranked = [Document(page_content=texts[i], metadata=all_docs[texts[i]].metadata) for i in sorted_indices]
    return reranked

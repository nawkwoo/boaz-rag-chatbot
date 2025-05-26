import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from config import DATA_PATH, SPARSE_INDEX_NAME, TOP_K_FINAL

load_dotenv()

# ✅ 전역 BM25 인코더 및 fit 여부 플래그
bm25_encoder = BM25Encoder()
_fit_done = False

def load_texts_from_jsonl(folder_path: str):
    texts = []
    metadatas = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                texts.append(record["text"])
                metadatas.append(record["metadata"])
    return texts, metadatas

def truncate_sparse_vector(sv, k=800):
    if not sv or "indices" not in sv or len(sv["indices"]) == 0:
        return {"indices": [], "values": []}
    if len(sv["indices"]) <= k:
        return sv
    topk = sorted(
        zip(sv["indices"], sv["values"]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:k]
    return {
        "indices": [i for i, _ in topk],
        "values": [v for _, v in topk]
    }

def search_documents(query: str, top_k: int = TOP_K_FINAL * 2) -> list[dict]:
    global _fit_done

    # ✅ 최초 1회만 BM25 학습
    if not _fit_done:
        texts, _ = load_texts_from_jsonl(DATA_PATH)
        bm25_encoder.fit(texts)
        _fit_done = True

    query_sparse = bm25_encoder.encode_queries([query])[0]
    query_sparse = truncate_sparse_vector(query_sparse)

    if len(query_sparse["indices"]) == 0:
        print("❌ Sparse 벡터가 비어 있어 검색 불가")
        return []

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(SPARSE_INDEX_NAME)

    result = index.query(
        vector=None,
        sparse_vector=query_sparse,
        top_k=top_k,
        include_metadata=True
    )

    hits = result.get("matches", [])
    return [
        {
            "text": hit["metadata"].get("text", ""),
            "metadata": hit["metadata"],
            "score": hit["score"]
        }
        for hit in hits
    ]

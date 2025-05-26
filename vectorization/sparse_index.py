# sparse_index.py
import os
import json
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec, Vector
from config import DATA_PATH, SPARSE_INDEX_NAME

load_dotenv()

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

# ✅ 검색 모듈에서 사용할 BM25Encoder를 전역으로 초기화 및 학습
texts_for_search, _ = load_texts_from_jsonl(DATA_PATH)
bm25_encoder = BM25Encoder()
bm25_encoder.fit(texts_for_search)

def truncate_sparse_vector(sv, k=800):
    # ✅ 빈 벡터는 빈 배열로 반환 (sparse vector 규칙)
    if not sv or "indices" not in sv or len(sv["indices"]) == 0:
        return {"indices": [], "values": []}  # 빈 sparse vector

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

def upload_sparse_index():
    print("✅ [SPARSE] 벡터 인덱싱 시작")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # ✅ 인덱스가 이미 존재하는지 확인만 (생성은 웹 콘솔에서)
    if SPARSE_INDEX_NAME not in pc.list_indexes().names():
        print(f"❌ 인덱스 '{SPARSE_INDEX_NAME}'가 존재하지 않습니다.")
        print("💡 Pinecone 웹 콘솔에서 Sparse 인덱스를 먼저 생성해주세요.")
        return

    texts, metadatas = load_texts_from_jsonl(DATA_PATH)
    encoder = BM25Encoder()
    encoder.fit(texts)  # ✅ fit 수행
    sparse_vectors = encoder.encode_documents(texts)
    sparse_vectors = [truncate_sparse_vector(vec) for vec in sparse_vectors]

    index = pc.Index(SPARSE_INDEX_NAME)
    to_upsert = []
    skipped = 0

    for i in range(len(texts)):
        vec = sparse_vectors[i]
        # ✅ 빈 벡터 완전 제거
        if len(vec["indices"]) == 0 or len(vec["values"]) == 0:
            skipped += 1
            continue
        to_upsert.append(Vector(id=f"id-{i}", sparse_values=vec, metadata=metadatas[i]))

    print(f"📊 업로드할 벡터: {len(to_upsert)}개 (건너뛴 벡터: {skipped}개)")

    BATCH_SIZE = 300
    success_count = 0

    for i in range(0, len(to_upsert), BATCH_SIZE):
        batch = to_upsert[i:i+BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            success_count += len(batch)
            print(f"✅ 배치 {i//BATCH_SIZE + 1} 완료 ({len(batch)}개)")
        except Exception as e:
            print(f"❌ 배치 {i//BATCH_SIZE + 1} 실패: {e}")

    print(f"✅ [SPARSE] 총 {success_count}개 벡터 업로드 완료")
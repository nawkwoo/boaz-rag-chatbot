import os
import json
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, Vector
from config import DATA_PATH, SPARSE_INDEX_NAME
from vectorization.utils import truncate_sparse_vector

load_dotenv()

def load_texts_from_jsonl(folder_path: str):
    texts, metadatas = [], []
    for fname in os.listdir(folder_path):
        if fname.endswith(".jsonl"):
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    texts.append(record["text"])
                    metadatas.append(record["metadata"])
    return texts, metadatas

def upload_sparse_index():
    print("✅ [SPARSE] 벡터 인덱싱 시작")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if SPARSE_INDEX_NAME not in pc.list_indexes().names():
        print(f"❌ 인덱스 '{SPARSE_INDEX_NAME}'가 존재하지 않습니다.")
        return

    texts, metadatas = load_texts_from_jsonl(DATA_PATH)
    encoder = BM25Encoder()
    encoder.fit(texts)
    sparse_vectors = encoder.encode_documents(texts)
    sparse_vectors = [truncate_sparse_vector(vec) for vec in sparse_vectors]

    index = pc.Index(SPARSE_INDEX_NAME)
    to_upsert = []
    id_to_text = {}
    skipped = 0

    for i, vec in enumerate(sparse_vectors):
        if not vec["indices"] or not vec["values"]:
            skipped += 1
            continue
        doc_id = f"sparse-{i}"
        to_upsert.append(Vector(id=doc_id, sparse_values=vec, metadata=metadatas[i]))
        id_to_text[doc_id] = texts[i]

    with open("id_to_text_sparse.json", "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)

    for i in range(0, len(to_upsert), 300):
        batch = to_upsert[i:i+300]
        try:
            index.upsert(vectors=batch)
            print(f"배치 {i//300 + 1} 완료 ({len(batch)}개)")
        except Exception as e:
            print(f"❌ 배치 {i//300 + 1} 실패: {e}")

    print(f"✅ [SPARSE] 총 {len(to_upsert)}개 벡터 업로드 완료 (건너뛴: {skipped})")

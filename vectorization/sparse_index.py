import os, json
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec, Vector

from config import DATA_PATH, BM25_INDEX_NAME

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

def truncate_sparse_vector(sv, k=800):
    if len(sv["indices"]) <= k:
        return sv
    topk = sorted(zip(sv["indices"], sv["values"]), key=lambda x: abs(x[1]), reverse=True)[:k]
    return {
        "indices": [i for i, _ in topk],
        "values": [v for _, v in topk]
    }

def upload_sparse_index():
    print("✅ [SPARSE] 벡터 인덱싱 시작")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if BM25_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=BM25_INDEX_NAME,
            dimension=1,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )

    texts, metadatas = load_texts_from_jsonl(DATA_PATH)
    encoder = BM25Encoder()
    encoder.fit(texts)
    sparse_vectors = encoder.encode_documents(texts)
    sparse_vectors = [truncate_sparse_vector(vec) for vec in sparse_vectors]

    index = pc.Index(BM25_INDEX_NAME)
    to_upsert = [
        Vector(id=f"id-{i}", sparse_values=sparse_vectors[i], metadata=metadatas[i])
        for i in range(len(texts))
    ]

    BATCH_SIZE = 100
    for i in range(0, len(to_upsert), BATCH_SIZE):
        batch = to_upsert[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)

    print(f"✅ [SPARSE] 총 {len(to_upsert)}개 벡터 업로드 완료")

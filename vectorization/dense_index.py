# dense_index.py

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec, Vector

from config import DATA_PATH, DENSE_INDEX_NAME, DENSE_MODEL_NAME, EMBEDDING_DIMENSION

load_dotenv()

def load_docs_from_jsonl(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                docs.append(Document(page_content=record["text"], metadata=record["metadata"]))
    return docs

def upload_dense_index():
    print("✅ [DENSE] 벡터 인덱싱 시작")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if DENSE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=DENSE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )

    docs = load_docs_from_jsonl(DATA_PATH)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    model = SentenceTransformer(DENSE_MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    to_upsert = [
        (f"id-{i}", vectors[i], metadatas[i])
        for i in range(len(texts))
    ]

    index = pc.Index(DENSE_INDEX_NAME)
    BATCH_SIZE = 100
    for i in range(0, len(to_upsert), BATCH_SIZE):
        batch = to_upsert[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)

    print(f"✅ [DENSE] 총 {len(to_upsert)}개 벡터 업로드 완료")
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec, Vector
from config import DATA_PATH, DENSE_INDEX_NAME, DENSE_MODEL_NAME

load_dotenv()

def load_docs_from_jsonl(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".jsonl"):
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    docs.append(Document(page_content=record["text"], metadata=record["metadata"]))
    return docs

def upload_dense_index():
    print("✅ [DENSE] 벡터 인덱싱 시작")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    model = SentenceTransformer(DENSE_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    if DENSE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=DENSE_INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1"))
        )

    docs = load_docs_from_jsonl(DATA_PATH)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    to_upsert = []
    id_to_text = {}

    for i in range(len(texts)):
        doc_id = f"dense-{i}"
        to_upsert.append(Vector(id=doc_id, values=vectors[i], metadata=metadatas[i]))
        id_to_text[doc_id] = texts[i]

    with open("id_to_text_dense.json", "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)

    index = pc.Index(DENSE_INDEX_NAME)
    for i in range(0, len(to_upsert), 100):
        batch = to_upsert[i:i+100]
        index.upsert(vectors=batch)
        print(f"배치 {i//100 + 1} 완료 ({len(batch)}개)")

    print(f"✅ [DENSE] 총 {len(to_upsert)}개 벡터 업로드 완료")

from vectorization.dense_index import load_docs_from_jsonl
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Vector
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("boaz-index2")

docs = load_docs_from_jsonl("data_with_meta/navercafe_study_fin.jsonl")
vectors = model.encode([doc.page_content for doc in docs]).tolist()

to_upsert = [
    Vector(id=f"test-{i}", values=vectors[i], metadata=docs[i].metadata)
    for i in range(len(docs))
]

index.upsert(vectors=to_upsert)
print("✅ 테스트 업로드 완료")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain.schema import Document

from config import SPARSE_INDEX_NAME, ID_TO_TEXT_PATH_SPARSE
from preprocess import load_documents

# 환경 변수 로드 (.env 파일에서 API 키, 환경명 등)
load_dotenv()

def create_and_upload_sparse_index():
    """
    문서 청킹 + BM25 Sparse 인코딩 → Pinecone 업로드 파이프라인

    1. 문서 로딩 및 청킹
    2. ID → 텍스트 매핑 저장
    3. BM25 벡터 인코딩
    4. Pinecone 인덱스 확인 및 연결
    5. 벡터 + 메타데이터 업로드
    """

    # 1. 문서 로딩
    all_docs: List[Document] = load_documents()
    if not all_docs:
        print("❌ 문서가 없습니다. 'data' 디렉토리를 확인하세요.")
        return
    print(f"✅ 문서 로딩 완료: 총 {len(all_docs)}개 문서 생성됨.")

    # 2. ID → 텍스트 매핑 저장 (JSON)
    texts = [doc.page_content for doc in all_docs]
    id_to_text = {str(i): text for i, text in enumerate(texts)}
    os.makedirs(os.path.dirname(ID_TO_TEXT_PATH_SPARSE), exist_ok=True)
    with open(ID_TO_TEXT_PATH_SPARSE, "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)
    print(f"✅ 로컬 매핑 저장 완료: {ID_TO_TEXT_PATH_SPARSE}")

    # 3. BM25 Sparse 인코딩
    encoder = BM25Encoder()
    encoder.fit(texts)  # 전체 말뭉치 기준으로 단어 빈도 계산
    sparse_vectors = encoder.encode_documents(texts)

    # 4. Pinecone 연결 및 인덱스 확인
    api_key = os.getenv("PINECONE_API_KEY")
    env_str = os.getenv("PINECONE_ENV", os.getenv("PINECONE_REGION", "us-east-1-aws"))
    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

    parts = env_str.split("-")
    cloud = parts[-1]
    region = "-".join(parts[:-1])

    pc = Pinecone(api_key=api_key)
    if SPARSE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"❌ 인덱스 '{SPARSE_INDEX_NAME}'가 존재하지 않습니다.")
    else:
        print(f"✅ 인덱스 '{SPARSE_INDEX_NAME}' 존재함")

    index = pc.Index(SPARSE_INDEX_NAME)

    # 5. Sparse 벡터 및 메타데이터 업로드 (100개씩 배치)
    batch_size = 100
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        ids = [str(i) for i in range(start, end)]
        sparse_vecs = sparse_vectors[start:end]
        metadatas = [doc.metadata for doc in all_docs[start:end]]

        upserts = [
            {
                "id": id_,
                "sparse_values": vec,
                "metadata": metadata
            }
            for id_, vec, metadata in zip(ids, sparse_vecs, metadatas)
        ]

        index.upsert(vectors=upserts)
        print(f"▶️ Upsert 완료: {ids[0]} ~ {ids[-1]}")

    print(f"모든 sparse vector 업로드 완료! 총 {total}개 문서")


if __name__ == "__main__":
    # 단독 실행 시 sparse 인덱스 생성 및 업로드 실행
    create_and_upload_sparse_index()

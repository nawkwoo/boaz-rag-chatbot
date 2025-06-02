import os
import json
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from config import DENSE_INDEX_NAME, DENSE_MODEL_NAME, ID_TO_TEXT_PATH
from preprocess import load_documents
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

# .env 파일에서 Pinecone API 키 및 환경 설정 로드
load_dotenv()

class SBERTEmbeddings:
    """
    SBERT 임베딩 모델 래퍼 클래스
    - 문서 전체 또는 단일 쿼리를 임베딩하여 벡터 반환
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        다수 문서를 임베딩하여 SBERT 벡터 리스트로 반환
        """
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 SBERT 벡터로 변환
        """
        vector = self.model.encode([text], show_progress_bar=False)[0]
        return vector.tolist()


def create_and_upload_vectorstore(
    index_name: str = DENSE_INDEX_NAME,
    model_name: str = DENSE_MODEL_NAME
) -> None:
    """
    Dense 벡터 인덱스를 Pinecone에 생성하고 문서를 업로드하는 파이프라인

    1. 문서 로딩 및 청킹
    2. SBERT 임베딩 수행
    3. 인덱스 존재 여부 확인 및 필요 시 생성
    4. id → 원문 매핑 저장
    5. 벡터 및 메타데이터 업로드 (batch 단위)
    """

    # 1. 문서 불러오기
    all_docs: List[Document] = load_documents()
    if not all_docs:
        print("❌ 문서가 없습니다. 'data' 디렉토리를 확인하세요.")
        return

    # 2. ID → 텍스트 매핑 저장 (JSON)
    id_to_text: dict = {str(i): doc.page_content for i, doc in enumerate(all_docs)}
    os.makedirs(os.path.dirname(ID_TO_TEXT_PATH), exist_ok=True)
    with open(ID_TO_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)
    print(f"✅ 로컬 매핑 파일 생성 완료: '{ID_TO_TEXT_PATH}' (총 {len(all_docs)}개 문서)")

    # 3. SBERT 모델 초기화
    embeddings = SBERTEmbeddings(model_name)

    # 4. Pinecone 연결 및 인덱스 준비
    api_key = os.getenv("PINECONE_API_KEY")
    env_str = os.getenv("PINECONE_ENV", os.getenv("PINECONE_REGION", "us-east-1-aws"))
    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

    # 환경 문자열 파싱: "us-east-1-aws" → region + cloud
    parts = env_str.split("-")
    cloud = parts[-1]
    region = "-".join(parts[:-1])

    pc = Pinecone(api_key=api_key)
    existing = pc.list_indexes().names()

    if index_name not in existing:
        print(f"➕ 인덱스 '{index_name}' 생성 중...")
        sample_text = all_docs[0].page_content[:100]
        dim = len(embeddings.embed_query(sample_text))
        spec = ServerlessSpec(cloud=cloud, region=region)
        pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)
    else:
        print(f"✅ 인덱스 '{index_name}' 이미 존재합니다. 이후 단계에서 upsert를 진행합니다.")

    index = pc.Index(index_name)

    # 5. 벡터 및 메타데이터 업로드
    texts = [doc.page_content for doc in all_docs]
    metadatas = [doc.metadata for doc in all_docs]
    ids = [str(i) for i in range(len(all_docs))]
    vectors = embeddings.embed_documents(texts)

    batch_size = 100
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_vectors = vectors[start:end]
        batch_meta = metadatas[start:end]

        upsert_items: List[Tuple[str, List[float], dict]] = list(zip(batch_ids, batch_vectors, batch_meta))
        index.upsert(vectors=upsert_items)
        print(f"▶️ Upsert 완료: {batch_ids[-1]} 까지 ({start+1} ~ {end})")

    print(f"✅ Dense 인덱스 '{index_name}' 업로드 완료: 총 {total}개 문서")


if __name__ == "__main__":
    # 단독 실행 시 벡터 업로드 수행
    create_and_upload_vectorstore()

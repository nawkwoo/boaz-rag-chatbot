from typing import Any, List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

import os
import json
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from config import SPARSE_INDEX_NAME, TOP_K, ID_TO_TEXT_PATH_SPARSE

# 환경 변수 로드 (.env에서 PINECONE_API_KEY, 환경명 등)
load_dotenv()


class SparsePineconeRetriever(BaseRetriever, BaseModel):
    """
    Pinecone Sparse 인덱스를 활용한 BM25 기반 LangChain 호환 Retriever

    - 질의를 Sparse 벡터로 인코딩하여 Pinecone에서 유사 문서 검색
    - 반환된 결과를 LangChain Document 리스트로 변환
    """

    index_name: str = Field(...)
    top_k: int = Field(...)
    encoder: Any = Field(...)
    index: Any = Field(...)
    id_to_text: dict = Field(...)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        질의를 BM25 Sparse 벡터로 인코딩하고 Pinecone에서 top-k 검색
        결과를 LangChain Document 형식으로 반환
        """
        query_vec = self.encoder.encode_queries([query])[0]

        results = self.index.query(
            top_k=self.top_k,
            sparse_vector=query_vec,
            include_metadata=True
        )

        docs = []
        for match in results.get("matches", []):
            doc_id = match["id"]
            text = self.id_to_text.get(doc_id, "")
            metadata = match.get("metadata", {})
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    class Config:
        arbitrary_types_allowed = True


def create_sparse_retriever(
    index_name: str = SPARSE_INDEX_NAME,
    top_k: int = TOP_K
) -> SparsePineconeRetriever:
    """
    SparsePineconeRetriever 인스턴스를 생성하는 헬퍼 함수

    1. 로컬에서 ID → 텍스트 매핑 로드
    2. 전체 텍스트에 대해 BM25Encoder 학습
    3. Pinecone 인덱스에 연결 후 Retriever 반환
    """
    with open(ID_TO_TEXT_PATH_SPARSE, "r", encoding="utf-8") as f:
        id_to_text = json.load(f)
    texts = list(id_to_text.values())

    encoder = BM25Encoder()
    encoder.fit(texts)  # 🔥 반드시 fit()을 호출하여 말뭉치 기반 인코딩 학습

    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENV", os.getenv("PINECONE_REGION", "us-east-1-aws"))
    pc = Pinecone(api_key=api_key, environment=env)
    index = pc.Index(index_name)

    return SparsePineconeRetriever(
        index_name=index_name,
        top_k=top_k,
        encoder=encoder,
        index=index,
        id_to_text=id_to_text
    )

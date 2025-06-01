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

# 환경 변수 로드
load_dotenv()

class SparsePineconeRetriever(BaseRetriever, BaseModel):
    index_name: str = Field(...)
    top_k: int = Field(...)
    encoder: Any = Field(...)
    index: Any = Field(...)
    id_to_text: dict = Field(...)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Query 인코딩
        query_vec = self.encoder.encode_queries([query])[0]

        # Pinecone에서 검색
        results = self.index.query(
            top_k=self.top_k,
            sparse_vector=query_vec,
            include_metadata=True
        )

        # 검색 결과를 Document로 변환
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
    # 문서 불러오기 (text만 리스트로)
    with open(ID_TO_TEXT_PATH_SPARSE, "r", encoding="utf-8") as f:
        id_to_text = json.load(f)
    texts = list(id_to_text.values())

    # BM25 인코더 초기화 후 학습
    encoder = BM25Encoder()
    encoder.fit(texts)  # 🔥 여기 필수!

    # Pinecone 연결
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


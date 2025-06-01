import os
import json
from typing import Any, List
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from langchain.schema import Document, BaseRetriever
from sentence_transformers import SentenceTransformer

from config import DENSE_INDEX_NAME, DENSE_MODEL_NAME, TOP_K, ID_TO_TEXT_PATH_DENSE

# .env 파일에서 환경변수 로드 (PINECONE_API_KEY, ENV 등)
load_dotenv()


class SBERTEmbeddings:
    """
    단일 쿼리를 SBERT 벡터로 변환하는 임베딩 래퍼 클래스
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class DensePineconeRetriever(BaseRetriever):
    """
    Pinecone에서 Dense 벡터 검색을 수행하는 LangChain 호환 Retriever 클래스
    """

    index: Any = None
    embeddings: Any = None
    id_to_text: dict = {}
    top_k: int = 0

    def __init__(self, index_name: str = DENSE_INDEX_NAME, top_k: int = TOP_K):
        super().__init__()

        # Pinecone API 초기화
        _api_key = os.getenv("PINECONE_API_KEY")
        _env = os.getenv("PINECONE_ENV", os.getenv("PINECONE_REGION", "us-east-1-aws"))
        if not _api_key:
            raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

        pc = PineconeClient(api_key=_api_key, environment=_env)
        self.index = pc.Index(index_name)

        self.embeddings = SBERTEmbeddings(DENSE_MODEL_NAME)
        self.top_k = top_k

        # ID → 원문 텍스트 매핑 불러오기
        if os.path.exists(ID_TO_TEXT_PATH_DENSE):
            with open(ID_TO_TEXT_PATH_DENSE, "r", encoding="utf-8") as f:
                self.id_to_text = json.load(f)
        else:
            self.id_to_text = {}
            print(f"[WARN] 매핑 파일을 찾을 수 없습니다: {ID_TO_TEXT_PATH_DENSE}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        쿼리를 벡터화한 뒤 Pinecone 검색 결과를 LangChain Document 형태로 반환
        """
        q_vec = self.embeddings.embed_query(query)

        results = self.index.query(
            vector=q_vec,
            top_k=self.top_k,
            include_metadata=True
        )

        docs: List[Document] = []
        for match in results.get("matches", []):
            doc_id = match.get("id", "")
            full_text = self.id_to_text.get(doc_id, "")
            meta = match.get("metadata", {})
            if full_text:
                docs.append(Document(page_content=full_text, metadata=meta))
        return docs


def create_dense_retriever(
    index_name: str = DENSE_INDEX_NAME,
    top_k: int = TOP_K
) -> DensePineconeRetriever:
    """
    외부에서 사용할 수 있도록 Retriever 인스턴스를 반환하는 헬퍼 함수
    """
    return DensePineconeRetriever(index_name=index_name, top_k=top_k)

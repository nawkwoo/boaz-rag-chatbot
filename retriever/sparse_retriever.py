# retriever/sparse_retriever.py

import os
import json
from typing import Any, List
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from pinecone_text.sparse import BM25Encoder
from langchain.schema import Document, BaseRetriever

from config import SPARSE_MODEL_NAME, ID_TO_TEXT_PATH_SPARSE

load_dotenv()


class SparsePineconeRetriever(BaseRetriever):
    index: Any = None
    encoder: Any = None
    id_to_text: dict = {}
    top_k: int = 0

    def __init__(self, index_name: str, top_k: int):
        super().__init__()

        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV", os.getenv("PINECONE_REGION", "us-east-1-aws"))
        if not api_key:
            raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

        pc = PineconeClient(api_key=api_key, environment=env)
        self.index = pc.Index(index_name)

        self.encoder = BM25Encoder(model_name=SPARSE_MODEL_NAME)
        self.top_k = top_k

        if os.path.exists(ID_TO_TEXT_PATH_SPARSE):
            with open(ID_TO_TEXT_PATH_SPARSE, "r", encoding="utf-8") as f:
                self.id_to_text = json.load(f)
        else:
            print(f"[WARN] 매핑 파일이 없습니다: {ID_TO_TEXT_PATH_SPARSE}")
            self.id_to_text = {}

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_sparse = self.encoder.encode_queries([query])[0]

        results = self.index.query(
            vector=query_sparse,
            top_k=self.top_k,
            include_metadata=True,
            namespace=""  # 선택적
        )

        docs: List[Document] = []
        for match in results.get("matches", []):
            doc_id = match.get("id", "")
            text = self.id_to_text.get(doc_id, "")
            meta = match.get("metadata", {})
            if text:
                docs.append(Document(page_content=text, metadata=meta))
        return docs

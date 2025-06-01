import os
import pandas as pd
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DATA_PATH


def load_documents() -> List[Document]:
    """
    data/ 폴더 내의 PDF 및 CSV 파일을 로드하고,
    텍스트를 chunk 단위로 분할하여 LangChain Document 리스트로 반환합니다.
    각 Document에는 출처 정보 및 청크 인덱스가 metadata로 포함됩니다.
    """
    all_docs: List[Document] = []

    # 텍스트 청킹 전략 정의
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,             # 최대 청크 길이
        chunk_overlap=100,          # 청크 간 중첩 길이
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for fname in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, fname)
        if not os.path.isfile(path):
            continue

        # PDF 파일 처리
        if fname.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(path)
                raw_docs = loader.load()
            except Exception as e:
                print(f"[❌ PDF 로딩 실패] {fname}: {e}")
                continue

            # 모든 페이지 내용을 이어붙인 후 chunking
            full_text = "\n".join([doc.page_content for doc in raw_docs])
            chunks = text_splitter.split_text(full_text)
            for idx, chunk in enumerate(chunks):
                metadata = {"source": fname, "chunk_index": idx}
                all_docs.append(Document(page_content=chunk, metadata=metadata))

        # CSV 파일 처리
        elif fname.lower().endswith(".csv"):
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except Exception as e:
                print(f"[❌ CSV 로딩 실패] {fname}: {e}")
                continue

            for idx, row in df.iterrows():
                # 각 행(row)을 하나의 문자열로 변환 후 chunking
                combined = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                chunks = text_splitter.split_text(combined)
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {
                        "source": fname,
                        "row_index": idx,
                        "chunk_index": chunk_idx
                    }
                    all_docs.append(Document(page_content=chunk, metadata=metadata))

        # 지원하지 않는 파일 형식은 무시
        else:
            continue

    print(f"✅ 문서 로딩 완료: 총 {len(all_docs)}개 문서 생성됨.")
    return all_docs

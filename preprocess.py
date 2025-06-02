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
        chunk_size=500,             # 청크 최대 길이
        chunk_overlap=100,          # 청크 간 중첩 길이
        length_function=len,        # 기본 길이 측정 함수
        separators=["\n\n", "\n", " ", ""]  # 청크 구분자 우선순위
    )

    # data/ 폴더 내 파일 순회
    for fname in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, fname)
        if not os.path.isfile(path):
            continue

        # PDF 파일 처리
        if fname.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(path)
                raw_docs = loader.load()  # 각 페이지별 Document 객체 리스트 반환
            except Exception as e:
                print(f"[❌ PDF 로딩 실패] {fname}: {e}")
                continue

            # 전체 페이지를 이어붙인 후 청크 단위로 분할
            full_text = "\n".join([doc.page_content for doc in raw_docs])
            chunks = text_splitter.split_text(full_text)
            for idx, chunk in enumerate(chunks):
                metadata = {
                    "source": fname,
                    "chunk_index": idx  # 몇 번째 청크인지 기록
                }
                all_docs.append(Document(page_content=chunk, metadata=metadata))

        # CSV 파일 처리
        elif fname.lower().endswith(".csv"):
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except Exception as e:
                print(f"[❌ CSV 로딩 실패] {fname}: {e}")
                continue

            for idx, row in df.iterrows():
                # 한 행을 "컬럼명: 값" 형식으로 합쳐 하나의 문자열로 변환
                combined = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                chunks = text_splitter.split_text(combined)
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {
                        "source": fname,
                        "row_index": idx,        # 원본 row 위치
                        "chunk_index": chunk_idx  # 해당 row 내 청크 순번
                    }
                    all_docs.append(Document(page_content=chunk, metadata=metadata))

        # 지원하지 않는 파일 형식은 무시
        else:
            continue

    print(f"✅ 문서 로딩 완료: 총 {len(all_docs)}개 문서 생성됨.")
    return all_docs

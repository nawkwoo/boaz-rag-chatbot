# 🤖 boaz-rag-chatbot

BOAZ 지원자들을 위한 **RAG(Retrieval-Augmented Generation)** 기반 챗봇입니다.  
지원자들이 자주 묻는 질문에 대해 의미 기반 검색 + 생성형 응답으로 정확하게 대응합니다.

---

## 📌 프로젝트 개요

BOAZ 리크루팅 기간마다 반복되는 FAQ 응답을 자동화하고자,  
크롤링된 PDF/CSV 데이터를 바탕으로 벡터 검색 + 생성형 모델을 결합한 챗봇을 구축했습니다.

- 사용자 질문을 벡터화하여 **Pinecone**에서 관련 문서 검색
- 검색된 문서를 기반으로 **LangChain + LLM**이 자연스럽게 답변 생성

---

## 🎯 목표

- BOAZ 공식 홈페이지에 배포 가능한 챗봇 구축
- 지원자의 다양한 질문에 정밀하게 대응
- 후속 기수도 유지보수 가능한 **모듈형 구조 설계**

## 🛠️ 프로젝트 구조
```
project_root/
├── .env # 🔒 API 키 및 환경변수 (.gitignore 포함)
├── config.py # 📋 모델/청크/인덱싱 전략 등 설정
├── requirements.txt # 📦 전체 의존성 목록

├── data/ # 📁 원본 CSV, PDF 데이터
├── data_with_meta/ # 🧩 메타데이터 + 청킹된 결과 저장

├── preprocess/ # 🔧 전처리 모듈
│ ├── loader.py # CSV, PDF 로딩
│ ├── splitter.py # 텍스트 청크 분할
│ ├── metadata.py # 파일 기반 메타데이터 생성
│ └── builder.py # 전체 전처리 파이프라인 실행

├── vectorization/ # 📌 벡터화 및 Pinecone 업로드
│ ├── upload_to_pinecone.py # config에 따라 dense/sparse 업로드 실행
│ ├── dense_index.py # SentenceTransformer 기반 dense 임베딩
│ ├── sparse_index.py # BM25 기반 sparse 벡터 생성
│ └── search.py # Pinecone 검색 함수

├── query/ # 🔍 검색 후 재정렬 처리
│ └── reranker.py # SBERT 또는 Cross-Encoder 기반 reranking
│ └── query_pipeline.py # 검색 + 재정렬 통합 처리

├── chatbot.py # LangChain 기반 응답 생성
├── chatbot_streamlit.py # Streamlit UI 인터페이스

├── run.py # 전처리 + 벡터 업로드 메인 실행
```

## 🚀 실행 방법
```bash
python -m venv venv
source venv\Scripts\activate (Windows)   # or venv/bin/activate(mac)

pip install -r requirements.txt

python run.py # 데이터 전처리 + 벡터 업로드
streamlit run chatbot_streamlit.py # 챗봇 실행 (웹 인터페이스)
```

## 👥 팀원

- 경재영, 김완철, 손관우, 정예린

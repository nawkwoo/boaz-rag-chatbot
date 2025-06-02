<<<<<<< HEAD
# boaz-rag-chatbot

🤖 RAG(Retrieval-Augmented Generation) 기반의 BOAZ FAQ 챗봇입니다.  
지원자들이 자주 묻는 질문에 대해 정확하고 동적인 응답을 제공합니다.

## 📌 프로젝트 소개

BOAZ 리크루팅 기간마다 반복되는 FAQ 응답을 자동화하고자,  
크롤링 + 벡터 검색 + 생성형 모델을 결합한 챗봇을 구축합니다.

사용자의 질문을 의미 기반으로 벡터화하여 관련 문서를 검색하고,  
그 문서를 기반으로 자연스러운 문장을 생성하여 응답합니다.

## 🎯 목표

=======
# 🤖 boaz-rag-chatbot
RAG(Retrieval-Augmented Generation) 기반의 BOAZ FAQ 챗봇입니다. 지원자들이 자주 묻는 질문에 대해 정확하고 동적인 응답을 제공합니다.

## 📌 프로젝트 소개
BOAZ 리크루팅 기간마다 반복되는 FAQ 응답을 자동화하고자,
크롤링 + 벡터 검색 + 생성형 모델을 결합한 챗봇을 구축합니다.

사용자의 질문을 의미 기반으로 벡터화하여 관련 문서를 검색하고,
그 문서를 기반으로 자연스러운 문장을 생성하여 응답합니다.

## 🎯 목표
>>>>>>> langchain
- BOAZ 공식 홈페이지에 챗봇 배포
- 지원자의 다양한 질문에 정확하게 대응
- 후속 기수에서도 유지보수 가능한 구조 설계

## 👥 팀원
<<<<<<< HEAD

- 경재영, 김완철, 손관우, 정예린
=======
경재영 · 김완철 · 손관우 · 정예린

## 🛠️ 실행 가이드

```bash
# 1. .env 파일에 Pinecone과 Gemini API 키 입력
# 예시 (.env)
# PINECONE_API_KEY=your_pinecone_key
# GEMINI_API_KEY=your_gemini_key
```
```bash
# 2. 가상환경 생성
python -m venv venv
```
```bash
# 3. 가상환경 활성화 (Windows 기준)
source venv/Scripts/activate
```
```bash
# 4. 패키지 설치
pip install -r requirements.txt
```
```bash
# 5. 벡터스토어 생성
python vectorstore/dense_uploader.py
python vectorstore/sparse_uploader.py
```
```bash
# 6. Streamlit 앱 실행
streamlit run app.py
```

## 📁 프로젝트 구조
```bash
boaz_rag/
├── app.py                     # Streamlit UI 실행
├── chain.py                   # Gemini LLM 및 QA 체인 정의
├── config.py                  # 전역 설정 (모델명, index명 등)
├── preprocess.py              # PDF/CSV 문서 로딩 및 청킹
├── retriever/                 # 벡터 검색기 정의
│   ├── dense_retriever.py     # SBERT 기반
│   ├── sparse_retriever.py    # BM25 기반
│   └── factory.py             # 설정 기반 retriever 선택
├── vectorstore/               # 인덱스 업로드 스크립트
│   ├── dense_uploader.py
│   └── sparse_uploader.py
├── data/                      # 원본 문서 저장 폴더
├── data_with_meta/            # 청크 + 매핑 JSON 저장소
│   └── id_to_text_dense.json
├── .env                       # 환경 변수 설정
├── requirements.txt           # 패키지 목록
```
>>>>>>> langchain

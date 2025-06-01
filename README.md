## 🛠️ 실행 순서

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
# 4. 필요한 패키지 설치
pip install -r requirements.txt
```
```bash
# 5. 벡터스토어 생성
python vectorstore.py
```
```bash
# 6. Streamlit 앱 실행
streamlit run app.py
```

## 📁 프로젝트 구조
```bash
boaz_rag/
├── app.py                  # Streamlit UI 실행 파일
├── chain.py                # Gemini LLM & QA 체인 정의
├── config.py               # 전역 설정 (모델명, top_k, 경로 등)
├── preprocess.py           # PDF/CSV 문서 로드 및 청킹
├── retriever.py            # Pinecone Dense Retriever 정의
├── vectorstore.py          # 문서 임베딩 및 Pinecone 업로드
├── data/                   # 원본 PDF/CSV 문서 폴더
├── data_with_meta/         # 가공된 청크 및 매핑 저장소
│   └── id_to_text_dense.json
├── .env                    # 환경변수 (API 키 등)
├── requirements.txt        # 의존 패키지 목록
```
import os
import streamlit as st
import logging
import random

from retriever.factory import create_retriever
from chain import GeminiLLM, build_qa_chain_with_rerank, run_qa_chain
from config import TOP_K

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 세션 상태 초기화 (질문/답변 히스토리 저장)
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_components():
    """
    Retriever와 Gemini 기반 LLM 인스턴스를 로드
    - 캐싱을 통해 반복 로딩 방지
    - 환경 변수로부터 Gemini API 키 확인
    """
    retriever = create_retriever()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
        st.stop()

    llm = GeminiLLM(api_key=gemini_api_key)
    return retriever, llm

def main():
    """
    Streamlit 앱의 메인 함수
    - 사용자 질의 입력
    - RAG 프로세스 실행
    - 결과 및 히스토리 출력
    """
    # 페이지 설정
    st.set_page_config(
        page_title="말해Boaz", 
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # 커스텀 CSS 스타일
    st.markdown("""
    <style>
    /* 전체 앱 스타일 */
    .stApp {
        max-width: 380px;
        margin: 0 auto;
        padding: 0.5rem;
        overflow-x: hidden;
    }
    
    /* 메인 컨테이너 */
    .main-container {
        background: linear-gradient(135deg, #a8b5ff 0%, #c8a8ff 100%);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        margin-bottom: 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* 제목 스타일 */
    .app-title {
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* 입력 영역 스타일 */
    .input-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin-top: -10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #a8b5ff 0%, #c8a8ff 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(168, 181, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 181, 255, 0.4);
    }
    
    /* 히스토리 카드 스타일 */
    .history-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #a8b5ff;
        width: 100%;
        box-sizing: border-box;
        overflow-wrap: break-word;
        word-wrap: break-word;
        word-break: break-word;
    }
    
    .question {
        color: #2c3e50;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        overflow-wrap: break-word;
        word-wrap: break-word;
        word-break: break-word;
    }
    
    .answer {
        color: #34495e;
        line-height: 1.6;
        font-size: 0.95rem;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        overflow-wrap: break-word;
        word-wrap: break-word;
        word-break: break-word;
        white-space: pre-wrap;
        max-width: 100%;
        box-sizing: border-box;
    }
    
    /* 로딩 스피너 */
    .stSpinner {
        text-align: center;
    }
    
    /* 텍스트 입력 필드 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #a8b5ff;
        box-shadow: 0 0 0 3px rgba(168, 181, 255, 0.1);
    }
    
    /* 히스토리 헤더 */
    .history-header {
        text-align: center;
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #a8b5ff;
    }
    
    /* 경고 메시지 */
    .stWarning {
        border-radius: 10px;
    }
    
    /* 빈 상태 메시지 */
    .empty-state {
        text-align: center;
        color: #7f8c8d;
        font-style: italic;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 메인 헤더
    st.markdown("""
    <div class="main-container">
        <div class="app-title">🤖 말해Boaz</div>
        <div class="app-subtitle">무엇이든 물어보세요!</div>
    </div>
    """, unsafe_allow_html=True)

    # 입력 영역
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    query = st.text_input("", placeholder="질문을 입력하세요...", key="query_input", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💬 질문하기"):
        if not query:
            st.warning("🤔 먼저 질문을 입력해주세요.")
        else:
            retriever, llm = load_components()

            # 랜덤하게 이름 선택
            names = ["재영이가", "예린이가", "완철이가", "관우가"]
            selected_name = random.choice(names)

            with st.spinner(f"🤖 {selected_name} 열심히 생각하고 있어요..."):
                # QA 체인 생성 및 실행
                qa_chain = build_qa_chain_with_rerank(llm, retriever, top_k=3)
                result = run_qa_chain(qa_chain, query)

                answer = result.get("result", "[결과 없음]")
                reranked_docs = result.get("source_documents", [])

                # 히스토리에 저장
                st.session_state.history.append((query, answer, reranked_docs))

    # 이전 질문/답변 히스토리 출력
    if st.session_state.history:
        st.markdown('<div class="history-header">📚 대화 기록</div>', unsafe_allow_html=True)

        for i, (q, a, docs) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
            <div class="history-card">
                <div class="question">❓ {q}</div>
                <div class="answer">{a}</div>
            </div>
            """, unsafe_allow_html=True)

            # 참조 문서는 콘솔에만 출력
            if docs:
                for doc in docs:
                    src = doc.metadata.get("source", "알 수 없음")
                    snippet = doc.page_content[:200].replace("\n", " ")
                    logger.info(f"[참조 문서] 파일: {src}, 내용 일부: {snippet}")
    else:
        st.markdown("""
        <div class="empty-state">
            아직 대화가 없습니다.<br>
            위에 질문을 입력해보세요! 🚀
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
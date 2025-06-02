import os
import streamlit as st
import logging

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
    st.set_page_config(page_title="Boaz RAG", page_icon="🤖")
    st.title("💬 Boaz 챗봇")

    query = st.text_input("질문을 입력하세요:", key="query_input")

    if st.button("질문하기"):
        if not query:
            st.warning("먼저 질문을 입력해주세요.")
        else:
            retriever, llm = load_components()

            with st.spinner("응답 생성 중..."):
                # QA 체인 생성 및 실행
                qa_chain = build_qa_chain_with_rerank(llm, retriever, top_k=3)
                result = run_qa_chain(qa_chain, query)

                answer = result.get("result", "[결과 없음]")
                reranked_docs = result.get("source_documents", [])

                # 히스토리에 저장
                st.session_state.history.append((query, answer, reranked_docs))

    # 이전 질문/답변 히스토리 출력
    if st.session_state.history:
        st.markdown("### History")

        for i, (q, a, docs) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}. {q}**")
            st.markdown(a)

            # 참조 문서는 콘솔에만 출력
            if docs:
                for doc in docs:
                    src = doc.metadata.get("source", "알 수 없음")
                    snippet = doc.page_content[:200].replace("\n", " ")
                    logger.info(f"[참조 문서] 파일: {src}, 내용 일부: {snippet}")

            st.markdown("---")

if __name__ == "__main__":
    main()

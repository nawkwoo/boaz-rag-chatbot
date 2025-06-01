import os
import streamlit as st
import logging

from config import TOP_K
from retriever import create_dense_retriever
from chain import GeminiLLM, build_qa_chain, run_qa_chain

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 세션 상태 초기화 (질문/답변 히스토리 저장)
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_components():
    """
    Retriever와 Gemini 기반 LLM을 로드하고, QA 체인을 생성합니다.
    Streamlit 캐시를 사용해 반복 로딩을 방지합니다.
    """
    retriever = create_dense_retriever()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
        st.stop()

    llm = GeminiLLM(api_key=gemini_api_key)
    qa_chain = build_qa_chain(llm, retriever)

    return qa_chain

def main():
    st.set_page_config(page_title="Boaz RAG", page_icon="🤖")
    st.title("💬 Boaz 챗봇")

    # 사용자 질문 입력
    query = st.text_input("질문을 입력하세요:", key="query_input")

    if st.button("질문하기"):
        if not query:
            st.warning("먼저 질문을 입력해주세요.")
        else:
            qa_chain = load_components()

            with st.spinner("응답 생성 중..."):
                result = run_qa_chain(qa_chain, query)
                answer = result.get("result", "")
                src_docs = result.get("source_documents", [])

                # 세션 기록 저장
                st.session_state.history.append((query, answer, src_docs))

    # 이전 Q&A 출력
    if st.session_state.history:
        st.markdown("### History")

        for i, (q, a, docs) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}. {q}**")
            st.markdown(a)

            # 콘솔에는 참조 문서 출력, 화면에는 미표시
            if docs:
                for doc in docs:
                    src = doc.metadata.get("source", "알 수 없음")
                    snippet = doc.page_content[:200].replace("\n", " ")
                    logger.info(f"[참조 문서] 파일: {src}, 내용 일부: {snippet}")

            st.markdown("---")

if __name__ == "__main__":
    main()

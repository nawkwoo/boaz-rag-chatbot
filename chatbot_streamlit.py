# chatbot_streamlit.py

import streamlit as st
from chatbot import get_final_answer

def main():
    st.set_page_config(page_title="Boaz RAG 챗봇", page_icon="🤖")
    st.title("💬 Boaz 챗봇")

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("질문을 입력하세요:", placeholder="예: 24기가 진행한 스터디 알려줘")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("질문하기")
    with col2:
        reset = st.button("초기화")

    if reset:
        st.session_state.history = []
        st.experimental_rerun()

    if ask and query:
        with st.spinner("검색 중..."):
            response = get_final_answer(query)
            st.session_state.history.append((query, response))

    if st.session_state.history:
        st.markdown("### 🧠 이전 질문/답변 기록")
        for i, (q, r) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}. {q}**")
            st.markdown(f"{r}")
            st.markdown("---")

if __name__ == "__main__":
    main()
import streamlit as st
from chatbot import get_final_answer

def main():
    st.set_page_config(page_title="Boaz RAG 챗봇", page_icon="🤖")
    st.title("💬 Boaz 챗봇")

    query = st.text_input("질문을 입력하세요:", placeholder="예: 24기가 진행한 스터디 알려줘")

    if st.button("질문하기") and query:
        with st.spinner("검색 중..."):
            response = get_final_answer(query)
            st.markdown("### ✅ 답변")
            st.write(response)

if __name__ == "__main__":
    main()

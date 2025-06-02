import os
import logging
from typing import Any, Mapping, Optional, List

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 로그 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiLLM(LLM):
    """
    Google Gemini API를 LangChain LLM 인터페이스로 감싼 커스텀 클래스
    """

    model_name: str = "gemini-2.0-flash"

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.callbacks = kwargs.get('callbacks', None)
        self.tags = kwargs.get('tags', None)
        self.verbose = kwargs.get('verbose', False)

        if model_name:
            self.model_name = model_name

        # Gemini API 구성
        genai.configure(api_key=api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        LangChain 내부에서 호출되는 메서드
        프롬프트를 받아 Gemini API로 응답을 생성
        """
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                logger.warning("Gemini API returned empty response")
                return "[응답 없음] 빈 응답을 반환했습니다."
        except ResourceExhausted as e:
            logger.warning(f"Gemini API quota exhausted: {e}")
            return "[할당량 초과] 잠시 후 다시 시도해주세요."
        except Exception as e:
            logger.error(f"Gemini 생성 오류: {e}", exc_info=True)
            return f"[LLM 호출 실패] {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "gemini"


# Cross-Encoder 기반 문서 재정렬
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def cross_encoder_rerank(query: str, docs: List[Any], top_k: int = 3) -> List[Any]:
    """
    Cross-Encoder 모델로 문서 relevance 점수 계산 후 재정렬
    """
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer.batch_encode_plus(pairs, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()

    scores = logits.tolist()
    reranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked[:top_k]


# 프롬프트 템플릿 정의
prompt_template = """
너는 'BOAZ'라는 이름의 빅데이터 연합동아리에 대해 안내하는 고도화된 전문 챗봇이야.
사용자는 이 동아리에 진지한 관심을 갖고 있으며, 질문에 대해 정확하고 신뢰도 높은 정보를 원해.

다음 지시사항을 반드시 지켜서 답변해 줘:

0. 사람 이름은 절대 제공하지 마.
1. "보아즈는 뭐하는 곳이야?"처럼 광범위한 질문에는 핵심 내용을 중심으로 300~400자 내외로 간결하게 정리해.
2. 복수의 결과가 있다면 최신 정보를 우선으로 알려줘.
3. 각 항목은 한두 문장으로 요약해. 너무 길게 설명하지 마.
4. "어떤게 있어?" 같은 질문엔 표 형태의 목록으로 깔끔하게 보여줘.
5. "어떤게 있었어?"처럼 과거를 묻는 질문에는 있다/없다, 년도, 기본 정보 중심으로 답해.
6. "OO을 추천해 주세요" 요청에는 근거 있는 간단한 추천을 해줘.
7. 개인정보 요청 시에는 "제공하기 어렵습니다"라고 명확히 거절해.
8. 답변 마지막엔 항상 “추가 문의가 필요하면 언제든 알려주세요.”라는 친근한 마무리 멘트를 넣어.
9. 동아리와 무관한 질문, 또는 정책상 답변 불가한 내용은 "죄송합니다, 답변 범위를 벗어납니다."라고 말해줘.

📘 규칙
- 답변은 반드시 한국어로 작성해.
- 제공된 참조 문서(context)의 내용을 최우선으로 고려해 답변해.
- 내용을 모를 경우, "정확한 정보가 없습니다. 공식 홈페이지 https://www.bigdataboaz.com 를 참고해주세요."라고 안내해.
- 민감하거나 추론성 높은 내용은 절대 작성하지 마. 이건 정말 중요해!

질문: {question}

참조 문서: {context}

답변 (한국어):
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_template,
)


def build_qa_chain_with_rerank(llm: LLM, retriever: Any, top_k: int = 3):
    """
    Cross-Encoder rerank가 통합된 LangChain QA 체인 구성
    """
    def rerank_retriever(query: str) -> List[Document]:
        initial_docs = retriever.get_relevant_documents(query)
        return cross_encoder_rerank(query, initial_docs, top_k=top_k)

    # LangChain의 RetrievalQA 구조를 커스터마이징
    class CustomQAChain:
        def invoke(self, inputs: dict):
            query = inputs["query"]
            docs = rerank_retriever(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            final_prompt = prompt.format(question=query, context=context)
            answer = llm(final_prompt)
            return {"result": answer, "source_documents": docs}

    return CustomQAChain()


def run_qa_chain(chain, query: str):
    """
    QA 체인을 실행하여 응답 및 참조 문서를 반환
    """
    try:
        if hasattr(chain, 'invoke'):
            result = chain.invoke({"query": query})
        else:
            result = chain({"query": query})
        return result
    except Exception as e:
        logger.error(f"QA Chain 실행 오류: {e}", exc_info=True)
        return {"result": f"[실행 실패] {str(e)}", "source_documents": []}

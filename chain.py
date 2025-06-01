import os
import logging
from typing import Any, Mapping, Optional, List

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiLLM(LLM):
    """
    Google Gemini 모델을 LangChain LLM 인터페이스로 감싼 클래스
    """

    model_name: str = "gemini-2.0-flash"

    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.callbacks = kwargs.get('callbacks', None)
        self.tags = kwargs.get('tags', None)
        self.verbose = kwargs.get('verbose', False)

        if model_name:
            self.model_name = model_name

        # Gemini API 키로 구성 설정
        genai.configure(api_key=api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        프롬프트를 입력받아 Gemini API로 응답 생성
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


def build_qa_chain(llm: LLM, retriever: Any):
    """
    주어진 LLM과 Retriever를 기반으로 QA 체인을 구성합니다.
    LangChain의 RetrievalQA를 사용하며, 한국어 사용자 응답에 최적화된 프롬프트 포함.
    """
    prompt_template = """
    너는 'BOAZ'라는 이름의 빅데이터 연합동아리에 대한 정보를 제공하는 고도화된 전문 챗봇이야.
    질문자는 이 동아리에 대해 진지하게 관심을 갖고 있는 지원자이므로, 질문에 정확하고 신뢰도 높은 정보를 바탕으로 답변해야 해!

    다음 규칙을 무조건 지켜줘!
    1. 답변은 반드시 한국어로 작성해.
    2. 제공된 참조 문서(context) 내용을 우선으로 고려해서서 답변해줘. 잘 모를 경우만 검색해해.
    3. 하나의 정리된 문단으로 응답해줘. 모르면 모른다고, 우리 https://www.bigdataboaz.com 공식 홈페이지 주소를 알려줘.
    4. 지원자가 혼동하지 않도록 핵심 내용을 중심으로 정리해서 설명해.
    5. 개인정보, 민감한 내용, 또는 추론성 정보는 절대 포함하지마! 이건 진짜 안 지키면 지구 멸망해.

    잘 부탁해. 너 꼼꼼하고 확실한 녀석이잖아.
    
    질문: {question}

    참조 문서: {context}

    답변 (한국어):
    """

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )

    # Retrieval 기반 QA 체인 구성
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain


def run_qa_chain(chain, query: str):
    """
    QA 체인을 실행하여 질의에 대한 응답 및 출처 문서 반환
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

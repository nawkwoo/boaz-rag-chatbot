from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from config import RERANKING_STRATEGY

# SBERT 모델 초기화
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-Encoder 모델 초기화
_ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_tokenizer = AutoTokenizer.from_pretrained(_ce_model_name)
_ce_model = AutoModelForSequenceClassification.from_pretrained(_ce_model_name)


def rerank_sbert(query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """문서들을 SBERT 기반으로 재정렬"""
    query_vec = _sbert_model.encode([query])[0]
    doc_vecs = _sbert_model.encode(docs)
    scores = cosine_similarity([query_vec], doc_vecs)[0]
    sorted_indices = scores.argsort()[::-1][:top_k]
    return [(docs[i], scores[i]) for i in sorted_indices]


def rerank_cross_encoder(query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """문서들을 Cross-Encoder 기반으로 재정렬"""
    pairs = [(query, doc) for doc in docs]
    inputs = _tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _ce_model(**inputs).logits.squeeze()
        if logits.dim() == 0:
            scores = [logits.item()]
        else:
            scores = logits.tolist()
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(docs[i], scores[i]) for i in sorted_indices]


def rerank(query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """config 기반으로 reranking 전략 실행"""
    if RERANKING_STRATEGY == "sbert":
        return rerank_sbert(query, docs, top_k)
    elif RERANKING_STRATEGY == "cross-encoder":
        return rerank_cross_encoder(query, docs, top_k)
    else:
        raise ValueError(f"지원되지 않는 reranking 전략: {RERANKING_STRATEGY}")

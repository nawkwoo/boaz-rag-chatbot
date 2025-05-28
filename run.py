### run.py

from preprocess.builder import process_all_documents
from vectorization.upload_to_pinecone import run_index_upload

print("🚀 [1/2] 전처리 시작")
process_all_documents("data", "data_with_meta")

print("🚀 [2/2] 벡터 인덱싱 실행")
run_index_upload()

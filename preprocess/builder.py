import os, json
from langchain.schema import Document
from preprocess.loader import load_document
from preprocess.splitter import get_splitter
from preprocess.metadata import extract_metadata_from_filename, enrich_metadata
from config import CHUNK_SIZE, CHUNK_OVERLAP, USE_METADATA

def process_all_documents(data_dir: str, output_dir: str):
    splitter = get_splitter()
    os.makedirs(output_dir, exist_ok=True)
    total_chunks = 0

    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        file_meta = extract_metadata_from_filename(path)
        file_meta["file_id"] = fname

        try:
            loaded_docs = load_document(path)
        except Exception as e:
            print(f"[❌ 로딩 실패] {fname}: {e}")
            continue

        for doc in loaded_docs:
            chunks = splitter.split_text(doc.page_content)
            output_path = os.path.join(output_dir, f"{fname}.jsonl")

            with open(output_path, "a", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks, start=1):
                    if USE_METADATA:
                        meta = enrich_metadata(
                            file_meta,
                            idx,
                            (idx - 1) * CHUNK_SIZE,
                            min(len(doc.page_content), idx * CHUNK_SIZE)
                        )
                    else:
                        meta = {}

                    json.dump({"text": chunk, "metadata": meta}, f, ensure_ascii=False)
                    f.write("\n")
                    total_chunks += 1

    print(f"✅ 총 {total_chunks}개의 청크가 '{output_dir}'에 저장되었습니다.")

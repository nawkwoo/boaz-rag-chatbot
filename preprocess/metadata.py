import os
import hashlib

def hash_fn(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def extract_metadata_from_filename(filepath: str) -> dict:
    try:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        return {
            "year": parts[1],
            "country": parts[2].split('.')[0],
        }
    except Exception as e:
        print(f"[메타데이터 추출 실패] {filepath}: {e}")
        return {"year": "unknown", "country": "unknown"}

def enrich_metadata(base_meta: dict, chunk_index: int, offset_start: int, offset_end: int) -> dict:
    return {
        **base_meta,
        "file_id": hash_fn(base_meta.get("file_id", "unknown")),
        "chunk_index": chunk_index,
        "offset_start": offset_start,
        "offset_end": offset_end
    }
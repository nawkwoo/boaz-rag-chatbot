import pandas as pd
import json
import os

def csv_to_jsonl(csv_path: str, jsonl_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].dropna().astype(str).tolist()

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            record = {
                "text": text,
                "metadata": {
                    "source": os.path.basename(csv_path),
                    "row_id": i
                }
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

# 예시 실행 (주석 해제하고 단독 실행 시 사용)
csv_to_jsonl("data/navercafe_study_fin.csv", "data_with_meta/navercafe_study_fin.jsonl")

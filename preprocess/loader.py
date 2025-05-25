from langchain_community.document_loaders import PyPDFLoader, CSVLoader

def load_document(filepath: str):
    """파일 확장자에 따라 적절한 문서 로더를 반환합니다."""
    if filepath.lower().endswith(".csv"):
        return CSVLoader(filepath, encoding="utf-8").load()
    elif filepath.lower().endswith(".pdf"):
        return PyPDFLoader(filepath).load()
    else:
        raise ValueError(f"[❌ 지원되지 않는 파일 형식] {filepath}")

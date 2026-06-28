import os

from langchain_naver import ClovaXEmbeddings


def get_embeddings() -> ClovaXEmbeddings:
    return ClovaXEmbeddings(
        model=os.environ.get("CLOVA_EMBEDDING_MODEL", "bge-m3"),
    )

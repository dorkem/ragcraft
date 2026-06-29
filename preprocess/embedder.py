import os

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=os.environ.get("EMBEDDING_MODEL", "dragonkue/BGE-m3-ko"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

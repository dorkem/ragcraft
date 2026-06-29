import os

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_vectorstore(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=os.environ.get("CHROMA_COLLECTION_NAME", "ragcraft"),
        embedding_function=embeddings,
        persist_directory=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
    )

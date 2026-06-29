import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def hash_pdf(pdf_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def is_duplicate(vectorstore: Chroma, file_hash: str) -> bool:
    result = vectorstore._collection.get(
        where={"file_hash": file_hash},
        limit=1,
        include=[],
    )
    return len(result["ids"]) > 0


def insert_documents(vectorstore: Chroma, chunks: list[Document], file_hash: str) -> None:
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
    vectorstore.add_documents(chunks)


def delete_by_source(vectorstore: Chroma, source: str) -> int:
    result = vectorstore._collection.get(
        where={"source": source},
        include=[],
    )
    ids = result["ids"]
    if ids:
        vectorstore._collection.delete(ids=ids)
    return len(ids)


def delete_by_hash(vectorstore: Chroma, file_hash: str) -> int:
    result = vectorstore._collection.get(
        where={"file_hash": file_hash},
        include=[],
    )
    ids = result["ids"]
    if ids:
        vectorstore._collection.delete(ids=ids)
    return len(ids)


def list_sources(vectorstore: Chroma) -> list[dict]:
    result = vectorstore._collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source": src,
                "file_hash": meta.get("file_hash", ""),
                "chunk_count": 0,
            }
        seen[src]["chunk_count"] += 1
    return list(seen.values())

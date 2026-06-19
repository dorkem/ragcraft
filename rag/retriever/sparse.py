import json
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def _load_parent_docs(chunk_dir: Path) -> List[Document]:
    """chunk JSON에서 parent 청크만 Document로 변환."""
    docs = []
    for chunk_file in sorted(chunk_dir.glob("*.json")):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        for c in chunks:
            if c.get("chunk_type") == "parent":
                docs.append(Document(
                    page_content=c["text"],
                    metadata=c["metadata"],
                ))
    return docs


def build_sparse_retriever(
    chunk_dir: Path,

    # ── 검색 수량 ────────────────────────────────────────────
    k: int = 10,
    # BM25에서 반환할 문서 수
    # Hybrid 조합 시 dense k와 동일하게 맞추는 것 권장

    # ── BM25 개요 ────────────────────────────────────────────
    # TF-IDF 개선판: 단어 빈도(TF) + 역문서빈도(IDF) 기반 키워드 매칭
    # Dense가 놓치는 exact-match 케이스를 보완
    #   예) "4-2-6조", "폴리프로필렌", 고유 조항 번호 등 정확한 용어
    #
    # rank_bm25 패키지 제공 알고리즘 (현재 Okapi 사용):
    # BM25Okapi (기본) : 표준 BM25, 일반적인 용도
    # BM25L            : 긴 문서 페널티 완화 (긴 페이지가 많을 때 유리)
    # BM25Plus         : 희귀 단어 가중치 강화 (전문용어 많을 때 유리)
    # → 알고리즘 변경 필요 시 BM25Retriever 대신 직접 구현 필요

) -> BM25Retriever:
    """BM25 키워드 기반 Sparse Retriever. parent(페이지 전체) 문서에서 검색."""
    docs = _load_parent_docs(chunk_dir)
    if not docs:
        raise ValueError(f"parent 청크 없음: {chunk_dir}")

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

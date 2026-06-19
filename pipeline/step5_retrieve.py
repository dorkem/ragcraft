"""
STEP 5: 검색 (Retrieve)
입력: ChromaDB + result/{date}/chunk/*.json (sparse/hybrid 전용)
출력: result/{date}/step5_retrieve.json

실행: python -m pipeline.step5_retrieve "질문"
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import settings
from rag.embeddings.clova_embedding import ClovaEmbedding
from rag.retriever.dense import FlatRetriever

# ── 설정 (여기서 변경) ──────────────────────────────────────
RETRIEVER_TYPE = "dense"      # "dense" | "sparse" | "hybrid"
K              = 10           # dense 검색 결과 수
SEARCH_TYPE    = "similarity" # "similarity" | "mmr"
SPARSE_K       = 10           # sparse 검색 결과 수
DENSE_WEIGHT   = 0.7          # hybrid dense 가중치
SPARSE_WEIGHT  = 0.3          # hybrid sparse 가중치
CHUNK_DATE     = None         # None = 오늘 날짜 / 예: "2026/06/19"
# ────────────────────────────────────────────────────────────



def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python -m pipeline.step5_retrieve \"질문\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    date = CHUNK_DATE or datetime.now().strftime("%Y/%m/%d")
    chunk_dir = Path(f"result/{date}/chunk")

    if RETRIEVER_TYPE in ("sparse", "hybrid") and not chunk_dir.exists():
        print(f"[ERROR] 청크 결과 없음: {chunk_dir}")
        print("  step2_chunk.py를 먼저 실행하거나 CHUNK_DATE를 확인하세요")
        sys.exit(1)

    print(f"\n[STEP 5] 검색 ({RETRIEVER_TYPE}) — \"{query}\"")

    if RETRIEVER_TYPE in ("dense", "hybrid"):
        embedding = ClovaEmbedding()
        vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        dense_retriever = FlatRetriever(
            vectorstore=vectorstore,
            k=K,
            search_type=SEARCH_TYPE,
        )

    if RETRIEVER_TYPE == "dense":
        retriever = dense_retriever

    elif RETRIEVER_TYPE == "sparse":
        from rag.retriever.sparse import build_sparse_retriever
        retriever = build_sparse_retriever(chunk_dir, k=SPARSE_K)

    else:  # hybrid
        from langchain.retrievers import EnsembleRetriever
        from rag.retriever.sparse import build_sparse_retriever
        sparse_retriever = build_sparse_retriever(chunk_dir, k=SPARSE_K)
        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
        )

    docs = retriever.invoke(query)

    W = 70
    print(f"\n{'=' * W}")
    print(f"  STEP 5  |  검색 ({RETRIEVER_TYPE.upper()})  |  k={K}  |  결과 {len(docs)}개")
    print(f"{'=' * W}")
    print(f"  질의: {query}")
    print(f"{'=' * W}")

    for i, d in enumerate(docs):
        filename = d.metadata.get("filename", "N/A")
        page = d.metadata.get("page_start", "?")
        print(f"\n[ {i+1} / {len(docs)} ]  {filename}  p.{page}")
        print(f"{'-' * W}")
        print(d.page_content)
        print(f"{'-' * W}")

    out_dir = Path(f"result/{date}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "query": query,
        "retriever_type": RETRIEVER_TYPE,
        "k": K,
        "chunk_date": date,
        "docs": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / "step5_retrieve.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 출력: {out_path}")


if __name__ == "__main__":
    main()

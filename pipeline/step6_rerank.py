"""
STEP 6: 리랭킹 (Rerank)
입력: pipeline/output/step5_retrieve.json
출력: pipeline/output/step6_rerank.json

실행: python -m pipeline.step6_rerank
"""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from rag.retriever.clova_reranker import ClovaReranker

# ── 설정 (여기서 변경) ──────────────────────────────────────
TOP_N = 5  # 리랭킹 후 반환할 문서 수 (권장: 3~5)
# ────────────────────────────────────────────────────────────

INPUT_PATH = Path("pipeline/output/step5_retrieve.json")
OUTPUT_DIR = Path("pipeline/output")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] 입력 파일 없음: {INPUT_PATH}")
        print("  step5_retrieve.py를 먼저 실행하세요")
        raise SystemExit(1)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    query = data["query"]
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data["docs"]]

    W = 70
    print(f"\n{'=' * W}")
    print(f"  STEP 6  |  리랭킹  |  {len(docs)}개 → top {TOP_N}")
    print(f"{'=' * W}")
    print(f"  질의: {query}")
    print(f"{'=' * W}")

    reranker = ClovaReranker(top_n=TOP_N)
    reranked = reranker.compress_documents(docs, query)

    for i, d in enumerate(reranked):
        filename = d.metadata.get("filename", "N/A")
        page = d.metadata.get("page_start", "?")
        score = d.metadata.get("relevance_score", 0)
        print(f"\n[ {i+1} / {len(reranked)} ]  {filename}  p.{page}  |  score={score:.3f}")
        print(f"{'-' * W}")
        print(d.page_content)
        print(f"{'-' * W}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "query": query,
        "top_n": TOP_N,
        "docs": [{"content": d.page_content, "metadata": d.metadata} for d in reranked],
        "timestamp": datetime.now().isoformat(),
    }
    out_path = OUTPUT_DIR / "step6_rerank.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 출력: {out_path}")


if __name__ == "__main__":
    main()

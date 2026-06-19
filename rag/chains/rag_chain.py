import json
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable

from core.config import settings
from rag.embeddings.clova_embedding import ClovaEmbedding
from rag.llm.hyperclova import HyperClovaLLM
from rag.retriever.clova_reranker import ClovaReranker
from rag.retriever.dense import ChildAwareParentRetriever

SYSTEM_PROMPT = """당신은 기술 지침 문서를 기반으로 답변하는 전문 어시스턴트입니다.
아래 참고 문서를 바탕으로 정확하고 간결하게 한국어로 답변하세요.
참고 문서에 없는 내용은 "제공된 문서에서 확인할 수 없습니다"라고 답변하세요.
답변 마지막에는 근거가 된 출처 파일명과 페이지를 반드시 표기하세요.
예) 출처: 조경시방서.pdf 3페이지

[참고 문서]
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _load_parent_store(chunk_dir: Path) -> InMemoryStore:
    """chunk JSON에서 parent 청크를 InMemoryStore에 로드."""
    store = InMemoryStore()
    for chunk_file in sorted(chunk_dir.glob("*.json")):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        parents = [c for c in chunks if c.get("chunk_type") == "parent"]
        if not parents:
            continue
        store.mset([
            (p["chunk_id"], Document(page_content=p["text"], metadata=p["metadata"]))
            for p in parents
        ])
    return store


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        filename = d.metadata.get("filename", "N/A")
        p_start = d.metadata.get("page_start")
        p_end = d.metadata.get("page_end")
        if p_start and p_end and p_start != p_end:
            page_info = f"{p_start}~{p_end}페이지"
        elif p_start:
            page_info = f"{p_start}페이지"
        else:
            page_info = "페이지 정보 없음"

        matched = d.metadata.get("matched_text", "")
        matched_line = f"[핵심 구절: {matched}]\n" if matched else ""

        parts.append(f"[출처: {filename} / {page_info}]\n{matched_line}{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(
    # ── 날짜 (chunk JSON 경로) ────────────────────────────────
    chunk_date: str = None,
    # result/{date}/chunk/ 에서 parent 문서 로드
    # None = 오늘 날짜 자동 사용 / 예: "2026/06/19"

    # ════════════════════════════════════════════════════════
    # RETRIEVER
    # ════════════════════════════════════════════════════════

    retriever_type: str = "dense",
    # "dense"  : 벡터 유사도 검색 (의미/맥락 기반)
    #            "인조잔디 설치 방법" → 직접 언급 없어도 관련 내용 검색
    # "sparse" : BM25 키워드 검색 (정확한 용어 기반)
    #            "4-2-6조", "폴리아미드" 같은 정확한 용어에 강함
    # "hybrid" : dense + sparse 혼합 (대부분의 경우 권장)
    #            두 방식 결과를 RRF(Reciprocal Rank Fusion)로 통합

    # ── Dense 파라미터 ────────────────────────────────────────
    search_type: str = "similarity",
    # "similarity" : 코사인 유사도 순 — 기본값
    # "mmr"        : Maximal Marginal Relevance
    #                비슷한 내용의 청크가 중복 반환될 때 사용

    k: int = 10,
    # ChromaDB에서 꺼낼 child 후보 수
    # 리랭커 있으면 크게(10~20), 없으면 최종 전달 수로(3~5)

    fetch_k: int = 20,
    # MMR 전용: 후보 풀 크기. k보다 커야 함

    lambda_mult: float = 0.5,
    # MMR 전용: 0.0(다양성 최대) ~ 1.0(관련성 최대)

    # ── Sparse 파라미터 ──────────────────────────────────────
    sparse_k: int = 10,
    # BM25 반환 문서 수 (dense k와 동일하게 맞추는 것 권장)

    # ── Hybrid 가중치 ─────────────────────────────────────────
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    # dense + sparse = 1.0 이 되도록 설정
    # 전문 용어 많은 문서 → sparse 비중 높이기 (0.4~0.5)
    # 의미 파악 중요한 문서 → dense 비중 높이기 (0.7~0.8)

    # ════════════════════════════════════════════════════════
    # RERANKER
    # ════════════════════════════════════════════════════════

    reranker_enabled: bool = None,
    # None  = settings.CLOVA_RERANKER_ENABLED 따름 (.env)
    # True  = 강제 활성화
    # False = 강제 비활성화 (검색 결과를 그대로 LLM에 전달)

    top_n: int = 3,
    # 리랭킹 후 LLM에 전달할 최종 문서 수
    # 많을수록 컨텍스트 풍부 ↔ 토큰 비용 증가 (권장: 3~5)

    # ════════════════════════════════════════════════════════
    # LLM
    # ════════════════════════════════════════════════════════

    temperature: float = 0.5,
    # 0.0 = 결정적 (항상 같은 답 / 사실 질의에 적합)
    # 1.0 = 창의적 (매번 다른 답)
    # RAG 권장: 0.1~0.5 (낮을수록 문서에 충실한 답변)

    max_tokens: int = 2048,
    # 최대 출력 토큰 수 (HCX-DASH-002 최대: 4096)

    top_p: float = 0.8,
    # Nucleus sampling: 상위 p% 확률 토큰에서만 샘플링
    # 0.5 = 보수적 / 0.9 = 다양한 표현
    # temperature와 동시 조절 시 한 쪽만 변경 권장

    repeat_penalty: float = 1.1,
    # 동일 문장/단어 반복 억제
    # 1.0 = 없음 / 1.5~2.0 = 강한 억제 (과하면 문체 어색해짐)

) -> Runnable:
    """RAG 체인 생성. 검색(dense/sparse/hybrid) → 리랭킹 → LLM 응답."""

    date = chunk_date or datetime.now().strftime("%Y/%m/%d")
    chunk_dir = Path(f"result/{date}/chunk")

    # ── 임베딩 & VectorStore ──────────────────────────────────
    embedding = ClovaEmbedding()
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    # ── Parent Store (chunk JSON → InMemoryStore) ─────────────
    docstore = _load_parent_store(chunk_dir) if chunk_dir.exists() else InMemoryStore()

    # ── Retriever 조립 ────────────────────────────────────────
    use_reranker = reranker_enabled if reranker_enabled is not None else settings.CLOVA_RERANKER_ENABLED

    if retriever_type == "sparse":
        from rag.retriever.sparse import build_sparse_retriever
        retriever = build_sparse_retriever(chunk_dir, k=sparse_k)

    elif retriever_type == "hybrid":
        from rag.retriever.sparse import build_sparse_retriever
        dense = ChildAwareParentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            k=k,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        sparse = build_sparse_retriever(chunk_dir, k=sparse_k)
        retriever = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[dense_weight, sparse_weight],
        )

    else:  # "dense" (기본값)
        retriever = ChildAwareParentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            k=k,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    # ── Reranker ──────────────────────────────────────────────
    if use_reranker:
        retriever = ContextualCompressionRetriever(
            base_compressor=ClovaReranker(top_n=top_n),
            base_retriever=retriever,
        )

    # ── LLM ──────────────────────────────────────────────────
    llm = HyperClovaLLM(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    # ── Chain 조립 (LCEL) ─────────────────────────────────────
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("질문: ").strip()
    print(f"\n질문: {query}\n")
    chain = build_chain()
    print(chain.invoke(query))

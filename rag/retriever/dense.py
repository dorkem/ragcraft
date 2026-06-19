from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class FlatRetriever(BaseRetriever):
    """ChromaDB 직접 검색. 플랫 청크를 그대로 반환."""

    vectorstore: object

    # ── 검색 수량 ────────────────────────────────────────────
    k: int = Field(default=5)
    # 반환할 청크 수
    # 리랭커 사용 시: 크게 설정(10~20) → 리랭커에서 추림
    # 리랭커 미사용 시: 최종 전달 수와 동일하게(3~5)

    # ── 검색 방식 ────────────────────────────────────────────
    search_type: str = Field(default="similarity")
    # "similarity" : 코사인 유사도 순 — 기본 시맨틱 검색
    # "mmr"        : Maximal Marginal Relevance
    #                비슷한 청크가 중복 반환될 때 사용

    # ── MMR 전용 파라미터 (search_type="mmr" 일 때만 적용) ────
    fetch_k: int = Field(default=20)
    # MMR 계산 전 먼저 꺼낼 후보 풀 크기 (k보다 커야 함)

    lambda_mult: float = Field(default=0.5)
    # 0.0 → 다양성 최대 / 1.0 → 관련성 최대
    # 권장: 0.3~0.7

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )
        return self.vectorstore.similarity_search(query, k=self.k)

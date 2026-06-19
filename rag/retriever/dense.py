from typing import List

from langchain.storage import InMemoryStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class ChildAwareParentRetriever(BaseRetriever):
    """child 청크로 검색 → parent(페이지 전체) 반환. Small-to-Big Retrieval.

    ChromaDB에는 child(작은 청크)만 저장 → 검색 정밀도 향상
    LLM에는 parent(페이지 전체) 전달 → 컨텍스트 풍부하게 유지
    """

    vectorstore: object
    docstore: InMemoryStore

    # ── 검색 수량 ────────────────────────────────────────────
    k: int = Field(default=10)
    # ChromaDB에서 꺼낼 child 후보 수
    # 리랭커 사용 시: 크게 설정(10~20) → 리랭커에서 추림
    # 리랭커 미사용 시: 최종 전달 수와 동일하게(3~5)

    # ── 검색 방식 ────────────────────────────────────────────
    search_type: str = Field(default="similarity")
    # "similarity" : 코사인 유사도 순 — 기본 시맨틱 검색
    # "mmr"        : Maximal Marginal Relevance
    #                비슷한 청크가 중복 반환될 때 사용
    #                관련성(relevance) + 다양성(diversity) 동시 고려

    # ── MMR 전용 파라미터 (search_type="mmr" 일 때만 적용) ────
    fetch_k: int = Field(default=20)
    # MMR 계산 전 먼저 꺼낼 후보 풀 크기 (k보다 커야 함)
    # 클수록 다양성 탐색 범위 넓어짐, 속도는 느려짐

    lambda_mult: float = Field(default=0.5)
    # 0.0 → 다양성 최대 (서로 다른 내용 선호)
    # 1.0 → 관련성 최대 (similarity 와 동일한 결과)
    # 권장: 0.3~0.7 / 도메인 특화 문서는 0.5~0.7

    id_key: str = Field(default="doc_id")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "mmr":
            child_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )
        else:
            child_docs = self.vectorstore.similarity_search(query, k=self.k)

        seen: dict[str, Document] = {}
        for child in child_docs:
            parent_id = child.metadata.get(self.id_key)
            if not parent_id or parent_id in seen:
                continue
            results = self.docstore.mget([parent_id])
            if results and results[0] is not None:
                parent = results[0]
                parent.metadata["matched_text"] = child.page_content
                seen[parent_id] = parent
        return list(seen.values())

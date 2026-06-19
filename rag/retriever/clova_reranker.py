from typing import Any, List, Optional, Sequence
import httpx
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import Field
from core.config import settings


class ClovaReranker(BaseDocumentCompressor):
    """CLOVA Studio Reranker LangChain 래퍼 (BaseDocumentCompressor)"""

    top_n: int = Field(default=5)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        if not documents:
            return []

        url = f"{settings.CLOVA_STUDIO_ENDPOINT}/testapp/v1/api-tools/reranker"
        payload = {
            "query": query,
            "documents": [{"id": str(i), "doc": doc.page_content} for i, doc in enumerate(documents)],
        }
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.CLOVA_STUDIO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"]["code"] != "20000":
            raise RuntimeError(f"CLOVA Reranker 오류: {data['status']}")

        cited = data["result"].get("citedDocuments", [])[: self.top_n]
        return [
            Document(
                page_content=r["doc"],
                metadata={
                    **documents[int(r["id"])].metadata,
                    "relevance_score": 1.0 / (i + 1),
                },
            )
            for i, r in enumerate(cited)
        ]

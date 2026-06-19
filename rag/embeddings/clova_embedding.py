from typing import List
import httpx
from langchain_core.embeddings import Embeddings
from core.config import settings


class ClovaEmbedding(Embeddings):
    """CLOVA Studio Embedding v2 (BGE-M3, 1024-dim) LangChain 래퍼"""

    def __init__(self):
        self._api_key = settings.CLOVA_STUDIO_API_KEY
        self._endpoint = settings.CLOVA_STUDIO_ENDPOINT
        self._model = settings.CLOVA_EMBEDDING_MODEL

    def _embed(self, text: str) -> List[float]:
        url = f"{self._endpoint}/testapp/v1/api-tools/embedding/v2"
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text, "model": self._model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"]["code"] != "20000":
            raise RuntimeError(f"CLOVA Embedding 오류: {data['status']}")
        return data["result"]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        result = []
        for text in texts:
            result.append(self._embed(text))
            time.sleep(0.05)  # rate limit
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

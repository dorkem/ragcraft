import os
import time
from typing import List

from langchain_naver import ClovaXEmbeddings
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_INTERVAL = float(os.environ.get("EMBED_INTERVAL", "0.5"))


class RateLimitedClovaXEmbeddings(ClovaXEmbeddings):

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(6),
    )
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, text in enumerate(texts):
            results.append(self.embed_query(text))
            if i < len(texts) - 1:
                time.sleep(_INTERVAL)
        return results


def get_embeddings() -> RateLimitedClovaXEmbeddings:
    return RateLimitedClovaXEmbeddings(
        model=os.environ.get("CLOVA_EMBEDDING_MODEL", "bge-m3"),
    )

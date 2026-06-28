import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class ClovaStudioEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.environ["CLOVASTUDIO_API_KEY"]
        endpoint = os.environ.get("CLOVA_STUDIO_ENDPOINT", "https://clovastudio.stream.ntruss.com")
        model = os.environ.get("CLOVA_EMBEDDING_MODEL", "bge-m3")
        self.url = f"{endpoint}/v1/api-tools/embedding/v2/{model}"

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()["result"]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

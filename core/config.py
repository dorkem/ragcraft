from typing import Literal
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # CLOVA Studio
    CLOVA_STUDIO_API_KEY: str
    CLOVA_STUDIO_ENDPOINT: str = "https://clovastudio.stream.ntruss.com"

    # 모델 선택
    HYPERCLOVA_MODEL: Literal["HCX-007", "HCX-005", "HCX-DASH-002", "HCX-003"] = "HCX-DASH-002"
    CLOVA_EMBEDDING_MODEL: str = "bge-m3"
    CLOVA_RERANKER_ENABLED: bool = True

    # 벡터 저장소 (ChromaDB)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "ragcraft"

    # 청킹
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "ragcraft"


settings = Settings()

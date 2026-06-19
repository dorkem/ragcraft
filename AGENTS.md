# 개발 지침 (AGENTS.md)

ragcraft 프로젝트의 상세 개발 규칙. 모든 섹션은 독립적으로 참조 가능하도록 작성됨.

---

## 목차

1. [아키텍처 & 데이터 흐름](#1-아키텍처--데이터-흐름)
2. [디렉토리 구조 & 컨벤션](#2-디렉토리-구조--컨벤션)
3. [LangChain 패턴](#3-langchain-패턴)
4. [HyperCLOVA X 통합](#4-hyperclova-x-통합)
5. [ChromaDB 통합](#5-chromadb-통합)
6. [환경 변수 관리](#6-환경-변수-관리)
7. [패키지 관리](#7-패키지-관리)
8. [코드 품질 & 스타일](#8-코드-품질--스타일)
9. [에러 처리](#9-에러-처리)
10. [테스트](#10-테스트)

---

## 1. 아키텍처 & 데이터 흐름

### 1.1 두 파이프라인의 분리

```
[적재 파이프라인] — pipeline/ — 1회성 배치, LangChain 불필요
  img/ 문서
    → step0_scan    : 새 파일 감지 (SHA256 중복 방지)
    → step1_parse   : PDF → 페이지별 텍스트 (pymupdf4llm)
    → step2_chunk   : Parent-Child 청킹 (page=parent, 300자=child)
    → step3_embed   : CLOVA Embedding v2 API 호출, 벡터 저장 JSON
    → step4_store   : ChromaDB에 child 청크 + 벡터 적재

[검색·응답 파이프라인] — rag/ — 실시간, LangChain LCEL
  질문
    → rag/retriever/dense.py   : ChildAwareParentRetriever (child 검색 → parent 반환)
       또는 sparse.py          : BM25 키워드 검색
       또는 hybrid             : EnsembleRetriever (dense + sparse RRF 결합)
    → rag/retriever/clova_reranker.py : CLOVA Reranker (citedDocuments 기반)
    → rag/llm/hyperclova.py    : HyperCLOVA X 답변 생성
    → 최종 답변
```

### 1.2 Parent-Child 구조

- **child**: 300자 분할 청크, ChromaDB에 벡터로 저장 (검색 정밀도)
- **parent**: 페이지 전체 텍스트, InMemoryStore에 로드 (LLM 컨텍스트 품질)
- child의 `doc_id` → parent의 `chunk_id` (`{filename}__page_{N}`)

```
ChromaDB: child 청크 (벡터 검색용)
  child.metadata["doc_id"] = "조경시방서.pdf__page_3"
                                        ↓
InMemoryStore: parent 청크 (LLM 전달용)
  key = "조경시방서.pdf__page_3"
  value = Document(page_content=페이지전체, metadata=...)
```

### 1.3 설계 원칙

- 적재(`pipeline/`)와 검색·응답(`rag/`)은 독립적으로 실행 가능해야 한다.
- `rag/`는 LangChain의 `Runnable` 인터페이스 기반으로 조합. 전역 상태 금지.
- `build_chain()`의 모든 파라미터는 주석으로 선택지와 기준을 명시한다.

---

## 2. 디렉토리 구조 & 컨벤션

### 2.1 모듈별 역할

| 디렉토리 | 역할 |
|----------|------|
| `core/` | 설정 로더 (`pydantic-settings`) |
| `pipeline/` | 적재 파이프라인 step0~step4 |
| `rag/embeddings/` | LangChain `Embeddings` 래퍼 |
| `rag/llm/` | LangChain `BaseChatModel` 래퍼 |
| `rag/retriever/` | Dense / Sparse / Reranker |
| `rag/chains/` | `build_chain()` — LCEL 조립 |

### 2.2 파일 네이밍

- 클래스: `PascalCase` (예: `HyperClovaLLM`, `ChildAwareParentRetriever`)
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`

### 2.3 `__init__.py` 규칙

공개 API만 re-export. 내부 구현 클래스는 노출하지 않는다.

---

## 3. LangChain 패턴

### 3.1 적용 범위

LangChain은 **검색·응답 파이프라인(`rag/`)에만** 사용한다.
적재 파이프라인(`pipeline/`)은 중간 결과 저장·재실행 가능성이 중요하므로 raw 코드로 유지한다.

단, `pipeline/step2_chunk.py`는 `RecursiveCharacterTextSplitter`를 사용한다 (텍스트 분할 유틸리티 목적).

### 3.2 체인 구성 (LCEL)

```python
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke("질문")
```

### 3.3 Retriever 선택 기준

| 종류 | 사용 시점 |
|------|----------|
| `dense` (기본) | 의미 기반 검색, 일반 질의 |
| `sparse` (BM25) | 조항 번호·고유명사 등 정확한 용어 검색 |
| `hybrid` | 위 두 가지가 모두 필요할 때 (권장) |
| `search_type="mmr"` | 유사 청크 중복 반환 문제 발생 시 |

### 3.4 Reranker

CLOVA Studio `/testapp/v1/api-tools/reranker` 엔드포인트는 **Grounded Generation** 방식:
- 요청: `{"query": "...", "documents": [{"id": "0", "doc": "..."}]}`
- 응답: `citedDocuments` 배열 (인용된 문서, 인용 순서 = 관련도 순)
- `rankingResults` 필드 없음 (구버전 스펙과 다름)

---

## 4. HyperCLOVA X 통합

### 4.1 래퍼 위치

`rag/llm/hyperclova.py` — `BaseChatModel` 상속.

### 4.2 파라미터

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `temperature` | 0.0~1.0 | RAG 권장: 0.1~0.5 |
| `max_tokens` | 1~4096 | 최대 출력 토큰 |
| `top_p` | 0.0~1.0 | Nucleus sampling |
| `repeat_penalty` | 1.0~2.0 | 반복 억제 |

### 4.3 API 호출 규칙

- 엔드포인트: `CLOVA_STUDIO_ENDPOINT/testapp/v1/chat-completions/{model}`
- 타임아웃: 60초, 재시도 없음 (상위에서 처리)
- API 키는 `core/config.py`에서만 로드

---

## 5. ChromaDB 통합

### 5.1 저장 구조

- **저장 대상**: child 청크만 (벡터 + 메타데이터)
- **parent 청크**: ChromaDB에 저장하지 않음 → `result/{date}/chunk/*.json`에서 InMemoryStore로 로드
- **ID 포맷**: `{filename}_{chunk_id}` (예: `조경시방서.pdf_조경시방서.pdf__child_0`)

### 5.2 메타데이터 스키마

```python
{
    "filename": "조경시방서_대외용.hwp.pdf",
    "file_path": "img/...",
    "file_hash": "sha256...",
    "file_size": 351877,
    "parser": "pymupdf4llm",
    "page_start": 3,
    "page_end": 3,
    "chunk_index": 2,
    "doc_id": "조경시방서_대외용.hwp.pdf__page_3",  # parent 참조 키
}
```

### 5.3 벡터 직접 적재

`step4_store.py`는 step3에서 계산된 벡터를 재사용하기 위해 `_collection.add()` 직접 호출.
`vectorstore.add_documents()` 사용 시 CLOVA Embedding API를 재호출하므로 사용하지 않는다.

---

## 6. 환경 변수 관리

### 6.1 로드 방식

`core/config.py`에서 `pydantic-settings`로 일괄 로드. 모듈별 `os.getenv()` 직접 호출 금지.

```python
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    CLOVA_STUDIO_API_KEY: str
    CLOVA_STUDIO_ENDPOINT: str = "https://clovastudio.stream.ntruss.com"
    HYPERCLOVA_MODEL: Literal["HCX-007", "HCX-005", "HCX-DASH-002", "HCX-003"] = "HCX-DASH-002"
    CLOVA_EMBEDDING_MODEL: str = "bge-m3"
    CLOVA_RERANKER_ENABLED: bool = True
    PARSER_TYPE: Literal["pymupdf4llm", "pymupdf", "tensorlake"] = "pymupdf4llm"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "ragcraft"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "ragcraft"

settings = Settings()
```

### 6.2 `.env.example` 동기화

`.env`에 새 키 추가 시 반드시 `.env.example`에도 추가.

---

## 7. 패키지 관리

### 7.1 핵심 의존성

```
langchain / langchain-core / langchain-community / langchain-text-splitters
langchain-chroma
chromadb
httpx
pydantic-settings
rank-bm25          # sparse retriever (BM25)
pymupdf / pymupdf4llm
streamlit
```

### 7.2 규칙

- `requirements.txt`로 관리, 버전은 `==`으로 고정.
- 패키지 추가 시 `requirements.txt`에 직접 추가 후 `pip install -r requirements.txt` 실행.

---

## 8. 코드 품질 & 스타일

### 8.1 타입 힌트

모든 함수에 파라미터/반환 타입 힌트 필수. `Any` 최소화.

### 8.2 주석 규칙

- `build_chain()` 파라미터: 선택지·범위·권장값 주석 필수.
- 일반 코드: WHY가 비자명할 때만 작성. WHAT 설명 금지.

### 8.3 임포트 순서

```python
# 1. 표준 라이브러리
# 2. 서드파티 (langchain, httpx 등)
# 3. 내부 모듈 (core, rag)
```

### 8.4 금지 패턴

- `print()` 디버깅 — pipeline 스크립트 외에서는 사용 금지
- 전역 가변 상태
- 인라인 임포트 (`rag/chains/rag_chain.py` 내 retriever_type 분기는 예외)

---

## 9. 에러 처리

```python
import logging
logger = logging.getLogger(__name__)

try:
    result = call_api()
except httpx.TimeoutException:
    logger.exception("CLOVA Studio API 타임아웃")
    raise
except httpx.HTTPStatusError as e:
    logger.exception("CLOVA Studio API 응답 오류 status=%s body=%s",
                     e.response.status_code, e.response.text)
    raise
```

커스텀 예외는 `core/exceptions.py`에 정의 (미구현 시 추가).

---

## 10. 테스트

### 10.1 구조

소스를 미러링: `rag/llm/hyperclova.py` → `tests/llm/test_hyperclova.py`

### 10.2 외부 의존성 격리

CLOVA Studio API / ChromaDB는 mock 처리. `pytest-mock` 사용.

### 10.3 실행

```bash
pytest tests/
pytest tests/llm/test_hyperclova.py
pytest -v
```

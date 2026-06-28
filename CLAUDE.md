# RAGcraft

건설 시방서 PDF를 파싱·청킹·임베딩하여 ChromaDB에 저장하고, HyperCLOVA X로 RAG 응답을 제공하는 파이프라인.

---

## 실행 방법

```bash
# 파일 선택 후 전처리 실행
python main.py
```

`main.py`가 `upload/` 목록을 출력하면 번호를 입력 → `preprocess.py <파일경로>`를 subprocess로 호출.

---

## 디렉토리 구조

```
ragcraft/
├── main.py                # 파일 선택 UI → preprocess.py 호출
├── preprocess.py          # 전처리 엔트리포인트 (단독 실행 가능)
├── preprocess/            # 전처리 모듈 (단독 실행 불가)
│   ├── parser.py          # PDF 파싱 (pymupdf4llm)
│   ├── chunker.py         # 청킹 (RecursiveCharacterTextSplitter)
│   ├── embedder.py        # 임베딩 (CLOVA Studio bge-m3)
│   └── vectorstore.py     # 벡터DB 저장 (ChromaDB)
└── upload/                # 입력 PDF 파일 위치
```

---

## 전처리 파이프라인

### 흐름

```
main.py → preprocess.py <path> → parse → chunk → embed → ChromaDB
```

1. **파싱** (`preprocess/parser.py`)
   - `pymupdf4llm.to_text(page_chunks=True)`로 페이지 단위 추출
   - 이미지 블록·구분선·연속 공백 제거
   - 메타데이터: `source`, `page_number`, `total_pages`, `file_path`

2. **청킹** (`preprocess/chunker.py`)
   - `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)`
   - 분리자: `["\n\n", "\n", ". ", " ", ""]`
   - 메타데이터에 `chunk_index`, `total_chunks` 추가
   - `chunk_index`는 파일 단위 0-based → 검색 후 앞뒤 청크 이어붙이기에 활용

3. **임베딩** (`preprocess/embedder.py`)
   - `langchain_naver.ClovaXEmbeddings(model="bge-m3")` 사용
   - 내부적으로 `/v1/openai` OpenAI-compatible 엔드포인트 호출 (1024차원)
   - `/v1/api-tools/embedding/v2/bge-m3` 경로는 404 — 사용 불가

4. **벡터DB 저장** (`preprocess/vectorstore.py`)
   - ChromaDB (`langchain_chroma.Chroma`)
   - 컬렉션: `CHROMA_COLLECTION_NAME` (기본값 `ragcraft`)
   - 저장 경로: `CHROMA_PERSIST_DIR` (기본값 `./chroma_db`)

---

## 환경변수 (.env)

| 키 | 설명 |
|---|---|
| `CLOVASTUDIO_API_KEY` | CLOVA Studio API 키 (`nv-` 접두어) |
| `CLOVA_STUDIO_ENDPOINT` | `https://clovastudio.stream.ntruss.com` |
| `CLOVA_EMBEDDING_MODEL` | `bge-m3` |
| `HYPERCLOVA_MODEL` | `HCX-003` |
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 |
| `CHROMA_COLLECTION_NAME` | ChromaDB 컬렉션 이름 |

---

## Retriever 설계 메모

`chunk_index` / `total_chunks` 메타데이터로 검색된 청크의 앞뒤 청크를 추가 조회해 이어붙인 뒤 LLM에 전달:

```python
hit = retrieved_chunk
source = hit.metadata["source"]
idx    = hit.metadata["chunk_index"]
# source 동일 + chunk_index == idx-1 또는 idx+1 인 청크를 추가 조회
```

---

## 기술 스택

| 역할 | 선택 |
|---|---|
| PDF 파싱 | pymupdf4llm |
| 청킹 | langchain-text-splitters |
| 임베딩 | langchain-naver `ClovaXEmbeddings` (bge-m3, 1024차원) |
| 벡터DB | ChromaDB (langchain-chroma) |
| LLM | HyperCLOVA X HCX-003 |
| 프레임워크 | LangChain |

# ragcraft

RAG 시스템 프로젝트. HyperCLOVA X(LLM) + ChromaDB(벡터 DB) + LangChain(검색·응답 파이프라인) 스택.

@AGENTS.md

---

## 문서 자동 업데이트 지침

> Claude는 대화 중 아래 조건이 발생하면 **즉시** 해당 파일을 업데이트한다. 대화가 끝날 때까지 미루지 않는다.

### 업데이트 트리거 → 대상 파일

| 발생 조건 | 업데이트 대상 |
|-----------|--------------|
| 새 디렉토리 / 파일 생성 | `CLAUDE.md` → 디렉토리 맵 |
| 아키텍처 결정 (모듈 추가, 설계 변경) | `AGENTS.md` § 1 아키텍처 |
| 새 LangChain 패턴 / 컨벤션 채택 | `AGENTS.md` § 3 LangChain 패턴 |
| HyperCLOVA X API 사용 방식 확정 | `AGENTS.md` § 4 HyperCLOVA X 통합 |
| ChromaDB 연결 / 쿼리 방식 확정 | `AGENTS.md` § 5 ChromaDB 통합 |
| 새 환경 변수 추가 | `CLAUDE.md` 핵심 환경 변수 + `AGENTS.md` § 6 |
| 패키지 추가 / 제거 | `AGENTS.md` § 7 패키지 관리 |
| 코드 스타일 규칙 변경 | `AGENTS.md` § 8 코드 품질 |
| 에러 처리 패턴 변경 | `AGENTS.md` § 9 에러 처리 |
| 테스트 전략 변경 | `AGENTS.md` § 10 테스트 |

### 업데이트 규칙

- 해당 섹션만 수정한다. 무관한 섹션은 건드리지 않는다.
- 결정 사항을 **구체적인 코드 예시**와 함께 반영한다.
- 과거 내용과 충돌 시 새 내용으로 교체하고, 이전 패턴은 삭제한다.
- 업데이트 후 사용자에게 별도 보고는 하지 않는다 — 그냥 한다.

---

## 프로젝트 스택

| 역할 | 기술 |
|------|------|
| LLM | HyperCLOVA X (CLOVA Studio API) |
| 벡터 저장소 | ChromaDB |
| RAG 파이프라인 | LangChain |
| 트레이싱 | LangSmith (선택) |
| UI | Streamlit |
| 런타임 | Python 3.10+ |

---

## 디렉토리 맵

```
ragcraft/
├── CLAUDE.md                          ← 지금 이 파일 (마스터 인덱스)
├── AGENTS.md                          ← 개발 지침 전체
├── README.md                          ← 프로젝트 소개 및 실행 가이드
├── .env / .env.example
├── requirements.txt
├── app.py                             ← Streamlit UI 진입점
├── core/
│   └── config.py                      ← pydantic-settings 설정 로더
├── img/                               ← 원본 문서 업로드 폴더 (git 제외)
│   └── complete_info/
│       └── processed.json             ← 처리 이력 (SHA256 기반, git 제외)
├── pipeline/                          ← 적재 + 검색 파이프라인
│   ├── step0_scan.py                  ← 스캔 & 중복 필터
│   ├── step1_parse.py                 ← 파싱 (pymupdf4llm / tensorlake)
│   ├── step2_chunk.py                 ← 플랫 청킹 (800자, 80자 오버랩)
│   ├── step3_embed.py                 ← 임베딩 (CLOVA Embedding v2)
│   ├── step4_store.py                 ← ChromaDB 저장
│   ├── step5_retrieve.py              ← 검색 (dense/sparse/hybrid)
│   ├── step6_rerank.py                ← CLOVA 리랭킹
│   ├── step7_generate.py              ← HyperCLOVA X 응답 생성
│   └── output/                        ← 스텝 간 중간 파일 (git 제외)
│       ├── step0_new_files.json
│       ├── step5_retrieve.json
│       ├── step6_rerank.json
│       └── step7_generate.json
├── rag/                               ← 검색·응답 파이프라인 (LangChain)
│   ├── embeddings/
│   │   └── clova_embedding.py         ← Embeddings 래퍼
│   ├── llm/
│   │   └── hyperclova.py              ← BaseChatModel 래퍼
│   ├── retriever/
│   │   ├── dense.py                   ← FlatRetriever (similarity/MMR)
│   │   ├── sparse.py                  ← BM25 Retriever
│   │   └── clova_reranker.py          ← BaseDocumentCompressor 래퍼
│   └── chains/
│       └── rag_chain.py               ← build_chain() — 모든 파라미터 노출
└── result/                            ← 날짜별 처리 결과 (git 제외)
    └── yyyy/mm/dd/
        ├── parse/{파일명}.json
        ├── chunk/{파일명}.json
        └── embed/{파일명}.json
```

---

## 빠른 참조

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py

# 적재 파이프라인 (날짜 지정)
python -m pipeline.step0_scan
python -m pipeline.step1_parse
python -m pipeline.step2_chunk
python -m pipeline.step3_embed
python -m pipeline.step4_store

# 검색·응답 단계별 실행 (step5~7 상수 수정 후 실행)
python -m pipeline.step5_retrieve "인조잔디 포장 방법은?"
python -m pipeline.step6_rerank
python -m pipeline.step7_generate

# RAG 체인 통합 테스트
python -m rag.chains.rag_chain 인조잔디 포장 방법은?
```

---

## 핵심 환경 변수 (`.env`)

| 키 | 기본값 | 용도 |
|----|--------|------|
| `CLOVA_STUDIO_API_KEY` | — | CLOVA Studio 인증 |
| `HYPERCLOVA_MODEL` | `HCX-DASH-002` | LLM 모델 선택 |
| `CLOVA_EMBEDDING_MODEL` | `bge-m3` | 임베딩 모델 |
| `CLOVA_RERANKER_ENABLED` | `true` | 리랭커 사용 여부 |
| `PARSER_TYPE` | `pymupdf4llm` | 문서 파서 선택 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 저장 경로 |
| `CHROMA_COLLECTION_NAME` | `ragcraft` | ChromaDB 컬렉션명 |

자세한 패턴과 개발 지침은 **AGENTS.md** 참조.

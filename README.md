# ragcraft

HyperCLOVA X + ChromaDB + LangChain 기반 RAG 시스템.
기술 지침 문서를 업로드하면 자연어로 질문하고 출처와 함께 답변을 받을 수 있습니다.

---

## 스택

| 역할 | 기술 |
|------|------|
| LLM | HyperCLOVA X (CLOVA Studio) |
| 임베딩 | CLOVA Embedding v2 (BGE-M3) |
| 리랭킹 | CLOVA Studio Reranker |
| 벡터 DB | ChromaDB |
| RAG 프레임워크 | LangChain |
| UI | Streamlit |

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 열어서 CLOVA_STUDIO_API_KEY 입력
```

### 3. 앱 실행

```bash
streamlit run app.py
```

---

## 파이프라인 구조

```
[적재]  img/ 문서 → step0~step4 → ChromaDB
[응답]  질문 → 검색(dense/sparse/hybrid) → 리랭킹 → HyperCLOVA X → 답변
```

### 적재 파이프라인 (pipeline/)

앱 사이드바의 **파이프라인 실행** 버튼으로 자동 실행되거나, 수동으로 각 단계 실행 가능:

```bash
python -m pipeline.step0_scan    # 새 파일 감지 (SHA256 중복 방지)
python -m pipeline.step1_parse   # PDF → 텍스트 변환
python -m pipeline.step2_chunk   # Parent-Child 청킹
python -m pipeline.step3_embed   # CLOVA 임베딩 생성
python -m pipeline.step4_store   # ChromaDB 저장
```

특정 날짜 지정:
```bash
python -m pipeline.step1_parse 2026/06/19
```

### 검색·응답 파이프라인 (rag/)

```bash
# RAG 체인 단독 테스트
python -m rag.chains.rag_chain 인조잔디 포장 방법은?
```

---

## 파라미터 튜닝

`rag/chains/rag_chain.py`의 `build_chain()` 함수에서 모든 파라미터를 조절할 수 있습니다.

```python
chain = build_chain(
    # Retriever
    retriever_type="hybrid",   # "dense" | "sparse" | "hybrid"
    search_type="mmr",         # "similarity" | "mmr"
    k=10,
    dense_weight=0.7,
    sparse_weight=0.3,

    # Reranker
    reranker_enabled=True,
    top_n=3,

    # LLM
    temperature=0.3,
    max_tokens=2048,
)
```

각 파라미터의 의미와 권장값은 `rag_chain.py` 주석 참조.

---

## 문서 추가 방법

1. `img/` 폴더에 PDF 파일 복사
2. 앱 사이드바 → **파이프라인 실행** 클릭
3. 완료 후 질문 입력

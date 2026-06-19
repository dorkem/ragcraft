"""
STEP 7: 응답 생성 (Generate)
입력: result/{date}/step6_rerank.json  (없으면 step5_retrieve.json fallback, 가장 최근 자동 탐색)
출력: result/{date}/step7_generate.json

실행: python -m pipeline.step7_generate
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm.hyperclova import HyperClovaLLM

# ── 설정 (여기서 변경) ──────────────────────────────────────
TEMPERATURE    = 0.3   # 0.0(결정적) ~ 1.0(창의적). RAG 권장: 0.1~0.5
MAX_TOKENS     = 2048  # 최대 출력 토큰
TOP_P          = 0.8   # Nucleus sampling
REPEAT_PENALTY = 1.1   # 반복 억제 (1.0=없음 / 1.5~2.0=강한 억제)
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 기술 지침 문서를 기반으로 답변하는 전문 어시스턴트입니다.
아래 참고 문서를 바탕으로 정확하고 간결하게 한국어로 답변하세요.
참고 문서에 없는 내용은 "제공된 문서에서 확인할 수 없습니다"라고 답변하세요.
답변 마지막에는 근거가 된 출처 파일명과 페이지를 반드시 표기하세요.
예) 출처: 조경시방서.pdf 3페이지

[참고 문서]
{context}"""


def _find_input() -> Path:
    """result/ 에서 step6_rerank.json → step5_retrieve.json 순으로 가장 최근 파일 반환."""
    for name in ("step6_rerank.json", "step5_retrieve.json"):
        candidates = sorted(
            Path("result").glob(f"*/*/*/{name}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        "step6_rerank.json / step5_retrieve.json을 result/ 에서 찾을 수 없습니다. "
        "step5_retrieve.py (또는 step6_rerank.py)를 먼저 실행하세요."
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        filename = d.metadata.get("filename", "N/A")
        p_start = d.metadata.get("page_start")
        p_end = d.metadata.get("page_end")
        if p_start and p_end and p_start != p_end:
            page_info = f"{p_start}~{p_end}페이지"
        elif p_start:
            page_info = f"{p_start}페이지"
        else:
            page_info = "페이지 정보 없음"
        parts.append(f"[출처: {filename} / {page_info}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    input_path = _find_input()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    query = data["query"]
    date = data["chunk_date"]
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data["docs"]]
    context = _format_docs(docs)

    W = 70
    print(f"\n{'=' * W}")
    print(f"  STEP 7  |  응답 생성  |  temp={TEMPERATURE}  |  from {input_path.name}")
    print(f"{'=' * W}")
    print(f"  질의: {query}")
    print(f"{'=' * W}")

    print(f"\n  참조 문서 ({len(docs)}개)")
    for i, d in enumerate(docs):
        filename = d.metadata.get("filename", "N/A")
        page = d.metadata.get("page_start", "?")
        print(f"  [{i+1}] {filename}  p.{page}")

    llm = HyperClovaLLM(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=query),
    ]
    result = llm._generate(messages)
    answer = result.generations[0].message.content
    token_usage = result.llm_output.get("token_usage", {})

    print(f"\n{'=' * W}")
    print(f"  답변")
    print(f"{'=' * W}\n")
    print(answer)
    print(f"\n{'=' * W}")
    print(f"  토큰: input={token_usage.get('input_tokens')}  output={token_usage.get('output_tokens')}")
    print(f"{'=' * W}")

    out_dir = Path(f"result/{date}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "query": query,
        "chunk_date": date,
        "answer": answer,
        "sources": [
            {
                "filename": d.metadata.get("filename"),
                "page_start": d.metadata.get("page_start"),
                "page_end": d.metadata.get("page_end"),
            }
            for d in docs
        ],
        "token_usage": token_usage,
        "from_step": input_path.name,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / "step7_generate.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 출력: {out_path}")


if __name__ == "__main__":
    main()

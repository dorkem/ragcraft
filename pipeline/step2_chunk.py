"""
STEP 2: 청킹 (Parent-Child / Small-to-Big)
입력: result/yyyy/mm/dd/parse/{파일명}.json  (페이지별 구조)
출력: result/yyyy/mm/dd/chunk/{파일명}.json

구조:
  parent_chunk: 1500자 (LLM 컨텍스트용, 검색 결과로 전달)
  child_chunks: 300자  (임베딩 & 검색용, 각 child가 parent_id 참조)

날짜 직접 지정: python -m pipeline.step2_chunk 2025/06/18
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TODAY = datetime.now().strftime("%Y/%m/%d")

PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def main():
    date = sys.argv[1] if len(sys.argv) >= 2 else TODAY
    src_dir = Path(f"result/{date}/parse")
    out_dir = Path(f"result/{date}/chunk")

    if not src_dir.exists():
        print(f"[ERROR] 파싱 결과 없음: {src_dir}")
        print("  step1_parse.py 를 먼저 실행하거나 날짜를 확인하세요")
        sys.exit(1)

    parse_files = sorted(src_dir.glob("*.json"))
    if not parse_files:
        print(f"[INFO] {src_dir} 에 파싱 결과가 없습니다")
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] Small-to-Big 청킹 시작 - {len(parse_files)}개 파일")
    print(f"  parent: {PARENT_CHUNK_SIZE}자 / child: {CHILD_CHUNK_SIZE}자")
    print(f"  출력: {out_dir}\n")

    for parse_file in parse_files:
        parsed = json.loads(parse_file.read_text(encoding="utf-8"))

        if "pages" in parsed:
            pages = parsed["pages"]
        else:
            pages = [{"page_number": 1, "content": parsed.get("content", "")}]

        base_meta = {
            **parsed["metadata"],
            "filename": parsed["filename"],
            "file_hash": parsed["file_hash"],
        }

        chunks = []
        child_id = 0

        # 페이지 경계를 절대 넘지 않음: parent = 1페이지, child = 그 안에서 분할
        for page in pages:
            page_num = page["page_number"]
            page_text = page["content"].strip()
            if not page_text:
                continue

            parent_id = f"{parsed['filename']}__page_{page_num}"

            # parent = 페이지 전체 (정확한 페이지 번호 보장)
            chunks.append({
                "chunk_id": parent_id,
                "chunk_type": "parent",
                "text": page_text,
                "metadata": {
                    **base_meta,
                    "page_start": page_num,
                    "page_end": page_num,
                },
            })

            # child = 페이지 내 고정 크기 분할
            child_texts = child_splitter.split_text(page_text)
            for c_text in child_texts:
                chunks.append({
                    "chunk_id": f"{parsed['filename']}__child_{child_id}",
                    "chunk_type": "child",
                    "doc_id": parent_id,
                    "text": c_text,
                    "metadata": {
                        **base_meta,
                        "chunk_index": child_id,
                        "doc_id": parent_id,
                        "page_start": page_num,
                        "page_end": page_num,
                    },
                })
                child_id += 1

        out_path = out_dir / parse_file.name
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

        parents = sum(1 for c in chunks if c["chunk_type"] == "parent")
        children = sum(1 for c in chunks if c["chunk_type"] == "child")
        print(f"  {parsed['filename']}")
        print(f"    parent {parents}개 / child {children}개 → {out_path.name}")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()

"""
STEP 2: 청킹 (Small-to-Big)
입력: result/yyyy/mm/dd/parse/{파일명}.json  (페이지별 구조)
출력: result/yyyy/mm/dd/chunk/{파일명}.json

구조:
  child_chunk: 300자 (임베딩 & 검색용, page_start/page_end로 정확한 페이지 안내)
  parent_chunk: child 5개 묶음 (LLM 컨텍스트용, child의 doc_id로 참조)

날짜 직접 지정: python -m pipeline.step2_chunk 2025/06/18
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TODAY = datetime.now().strftime("%Y/%m/%d")

CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30
CHILDREN_PER_PARENT = 5
MIN_CHILD_LEN = 80

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
    print(f"  child: {CHILD_CHUNK_SIZE}자 / parent: child {CHILDREN_PER_PARENT}개 묶음")
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
        filename = parsed["filename"]

        # 페이지별로 child 분할 → page 메타 보존
        all_children: list[dict] = []
        for page in pages:
            page_num = page["page_number"]
            page_text = page["content"].strip()
            if not page_text:
                continue
            for text in child_splitter.split_text(page_text):
                if (all_children
                        and len(text) < MIN_CHILD_LEN
                        and all_children[-1]["page"] == page_num):
                    all_children[-1]["text"] += " " + text
                else:
                    all_children.append({"text": text, "page": page_num})

        chunks = []

        # parent: child CHILDREN_PER_PARENT개 묶음
        for parent_idx, start in enumerate(range(0, len(all_children), CHILDREN_PER_PARENT)):
            group = all_children[start: start + CHILDREN_PER_PARENT]
            parent_id = f"{filename}__parent_{parent_idx}"
            parent_text = "\n".join(c["text"] for c in group)
            chunks.append({
                "chunk_id": parent_id,
                "chunk_type": "parent",
                "text": parent_text,
                "metadata": {
                    **base_meta,
                    "page_start": group[0]["page"],
                    "page_end": group[-1]["page"],
                },
            })

        # child: 검색 hit 시 page_start로 정확한 페이지 안내
        for child_idx, child_info in enumerate(all_children):
            parent_id = f"{filename}__parent_{child_idx // CHILDREN_PER_PARENT}"
            chunks.append({
                "chunk_id": f"{filename}__child_{child_idx}",
                "chunk_type": "child",
                "doc_id": parent_id,
                "text": child_info["text"],
                "metadata": {
                    **base_meta,
                    "chunk_index": child_idx,
                    "doc_id": parent_id,
                    "page_start": child_info["page"],
                    "page_end": child_info["page"],
                },
            })

        out_path = out_dir / parse_file.name
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

        parents = sum(1 for c in chunks if c["chunk_type"] == "parent")
        children = sum(1 for c in chunks if c["chunk_type"] == "child")
        print(f"  {filename}")
        print(f"    parent {parents}개 / child {children}개 → {out_path.name}")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()

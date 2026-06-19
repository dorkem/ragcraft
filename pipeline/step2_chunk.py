"""
STEP 2: 청킹 (Flat Chunking)
입력: result/yyyy/mm/dd/parse/{파일명}.json
출력: result/yyyy/mm/dd/chunk/{파일명}.json

구조:
  페이지별로 800자 청크 (오버랩 80자). page_start/page_end 메타데이터 보존.

날짜 직접 지정: python -m pipeline.step2_chunk 2025/06/18
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TODAY = datetime.now().strftime("%Y/%m/%d")

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80
MIN_CHUNK_LEN = 100  # 이보다 짧은 잔여 청크는 버림

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def main() -> None:
    date = sys.argv[1] if len(sys.argv) >= 2 else TODAY
    src_dir = Path(f"result/{date}/parse")
    out_dir = Path(f"result/{date}/chunk")

    if not src_dir.exists():
        print(f"[ERROR] 파싱 결과 없음: {src_dir}")
        print("  step1_parse.py를 먼저 실행하거나 날짜를 확인하세요")
        sys.exit(1)

    parse_files = sorted(src_dir.glob("*.json"))
    if not parse_files:
        print(f"[INFO] {src_dir} 에 파싱 결과가 없습니다")
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] 플랫 청킹 시작 - {len(parse_files)}개 파일")
    print(f"  chunk_size={CHUNK_SIZE} / overlap={CHUNK_OVERLAP}")
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

        chunks = []
        chunk_idx = 0

        for page in pages:
            page_num = page["page_number"]
            page_text = page["content"].strip()
            if not page_text:
                continue

            for text in splitter.split_text(page_text):
                if len(text.strip()) < MIN_CHUNK_LEN:
                    continue
                chunks.append({
                    "chunk_id": f"{filename}__chunk_{chunk_idx}",
                    "text": text,
                    "metadata": {
                        **base_meta,
                        "chunk_index": chunk_idx,
                        "page_start": page_num,
                        "page_end": page_num,
                    },
                })
                chunk_idx += 1

        out_path = out_dir / parse_file.name
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {filename} → {len(chunks)}개 청크 → {out_path.name}")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()

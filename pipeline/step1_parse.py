"""
STEP 1: 문서 파싱 (pymupdf4llm)
입력: pipeline/output/step0_new_files.json
출력: result/yyyy/mm/dd/parse/{파일명}.json  (페이지별 구조로 저장)
      img/complete_info/processed.json       (처리 이력 업데이트)

저장 구조:
{
  "filename": "...",
  "pages": [
    {"page_number": 1, "content": "..."},
    {"page_number": 2, "content": "..."},
    ...
  ]
}
"""
import json
import sys
from datetime import datetime
from pathlib import Path

INPUT_FILE = Path("pipeline/output/step0_new_files.json")
PROCESSED_LOG = Path("img/complete_info/processed.json")
TODAY = datetime.now().strftime("%Y/%m/%d")


def result_path(filename: str) -> Path:
    stem = Path(filename).stem
    return Path(f"result/{TODAY}/parse/{stem}.json")


def parse(file_path: Path) -> list[dict]:
    import pymupdf4llm
    chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    return [
        {"page_number": chunk["metadata"]["page_number"], "content": chunk["text"]}
        for chunk in chunks
        if chunk["text"].strip()
    ]


def update_processed_log(file_info: dict, total_chars: int, output_path: Path):
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = {"processed": {}}
    if PROCESSED_LOG.exists():
        log = json.loads(PROCESSED_LOG.read_text(encoding="utf-8"))

    log["processed"][file_info["file_hash"]] = {
        "filename": file_info["filename"],
        "file_path": file_info["file_path"],
        "processed_at": datetime.now().isoformat(),
        "file_size": file_info["file_size"],
        "total_chars": total_chars,
        "result_path": str(output_path),
    }
    PROCESSED_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if not INPUT_FILE.exists():
        print("[ERROR] step0_new_files.json 없음 - step0_scan.py 먼저 실행하세요")
        sys.exit(1)

    new_files = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    if not new_files:
        print("[INFO] 파싱할 신규 문서 없음")
        sys.exit(0)

    print(f"\n[STEP 1] 파싱 시작 - {len(new_files)}개 / 파서: pymupdf4llm")
    print(f"출력 경로: result/{TODAY}/parse/\n")

    success, errors = 0, []

    for info in new_files:
        file_path = Path(info["file_path"])
        out_path = result_path(info["filename"])
        print(f"  파싱: {info['filename']}")

        try:
            pages = parse(file_path)

            total_chars = sum(len(p["content"]) for p in pages)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "filename": info["filename"],
                "file_hash": info["file_hash"],
                "parsed_at": datetime.now().isoformat(),
                "total_pages": len(pages),
                "total_chars": total_chars,
                "pages": pages,
                "metadata": {
                    "file_path": info["file_path"],
                    "file_size": info["file_size"],
                    "parser": "pymupdf4llm",
                },
            }
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            update_processed_log(info, total_chars, out_path)

            print(f"  완료: {len(pages)}페이지 / {total_chars:,}글자 → {out_path}")
            success += 1

        except Exception as e:
            errors.append({"filename": info["filename"], "error": str(e)})
            print(f"  ERROR: {e}")

    print(f"\n성공: {success} / 실패: {len(errors)}")
    if errors:
        for e in errors:
            print(f"  - {e['filename']}: {e['error']}")


if __name__ == "__main__":
    main()

"""
STEP 0: 문서 스캔 & 중복 필터
- img/ 폴더 전체 스캔
- img/complete_info/processed.json 으로 처리 이력 확인 (SHA256 기반)
- 미처리 파일 목록만 pipeline/output/step0_new_files.json 에 저장
- 파싱/청킹과 무관하게 단독 실행 가능
"""
import hashlib
import json
import sys
from pathlib import Path

IMG_DIR = Path("img")
PROCESSED_LOG = IMG_DIR / "complete_info" / "processed.json"
OUTPUT_FILE = Path("pipeline/output/step0_new_files.json")

SUPPORTED_EXTENSIONS = {".pdf"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_processed_log() -> dict:
    if PROCESSED_LOG.exists():
        return json.loads(PROCESSED_LOG.read_text(encoding="utf-8"))
    return {"processed": {}}


def main():
    if not IMG_DIR.exists():
        print(f"[ERROR] img/ 폴더가 없습니다 ({IMG_DIR.resolve()})")
        sys.exit(1)

    all_files = sorted(
        f for f in IMG_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_files:
        print("[INFO] img/ 에 처리할 문서가 없습니다")
        sys.exit(0)

    print(f"\n[STEP 0] 스캔 완료 - 총 {len(all_files)}개 발견")

    log = load_processed_log()
    processed_hashes = set(log["processed"].keys())

    new_files, skip_count = [], 0
    for f in all_files:
        h = file_hash(f)
        if h in processed_hashes:
            prev = log["processed"][h]
            print(f"  SKIP  {f.name}  (처리일: {prev['processed_at'][:10]})")
            skip_count += 1
        else:
            new_files.append({
                "filename": f.name,
                "file_path": str(f),
                "file_hash": h,
                "file_size": f.stat().st_size,
            })
            print(f"  NEW   {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(new_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n신규: {len(new_files)}개 / 건너뜀: {skip_count}개")
    print(f"저장 → {OUTPUT_FILE}")

    if not new_files:
        print("[INFO] 새 문서 없음 - 이후 스텝 불필요")


if __name__ == "__main__":
    main()

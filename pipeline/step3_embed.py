"""
STEP 3: 임베딩 (CLOVA Studio Embedding v2 - BGE-M3, 1024-dim)
입력: result/yyyy/mm/dd/chunk/{파일명}.json
출력: result/yyyy/mm/dd/embed/{파일명}.json

날짜 직접 지정: python -m pipeline.step3_embed 2025/06/18
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm

TODAY = datetime.now().strftime("%Y/%m/%d")


def embed_text(text: str, api_key: str, endpoint: str, model: str) -> list[float]:
    url = f"{endpoint}/testapp/v1/api-tools/embedding/v2"
    for attempt in range(5):
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"text": text, "model": model},
            timeout=30,
        )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 0))
            wait = retry_after if retry_after > 0 else 60
            print(f"\n  [429] rate limit - {wait}초 대기 후 재시도 ({attempt+1}/5)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if data["status"]["code"] != "20000":
            raise RuntimeError(f"CLOVA API 오류: {data['status']}")
        return data["result"]["embedding"]
    raise RuntimeError("임베딩 실패: 재시도 횟수 초과 (429)")


def main():
    date = sys.argv[1] if len(sys.argv) >= 2 else TODAY
    chunk_dir = Path(f"result/{date}/chunk")
    embed_dir = Path(f"result/{date}/embed")

    if not chunk_dir.exists():
        print(f"[ERROR] 청킹 결과 없음: {chunk_dir}")
        print("  step2_chunk.py 를 먼저 실행하거나 날짜를 확인하세요")
        sys.exit(1)

    chunk_files = sorted(chunk_dir.glob("*.json"))
    if not chunk_files:
        print(f"[INFO] {chunk_dir} 에 청킹 결과가 없습니다")
        sys.exit(0)

    from core.config import settings

    embed_dir.mkdir(parents=True, exist_ok=True)
    total_ok, total_err = 0, 0

    for chunk_file in chunk_files:
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        print(f"\n[STEP 3] {chunk_file.name} - {len(chunks)}개 청크 임베딩")

        embedded, errors = [], []
        for chunk in tqdm(chunks, desc=f"  {chunk_file.stem}"):
            try:
                vector = embed_text(
                    text=chunk["text"],
                    api_key=settings.CLOVA_STUDIO_API_KEY,
                    endpoint=settings.CLOVA_STUDIO_ENDPOINT,
                    model=settings.CLOVA_EMBEDDING_MODEL,
                )
                embedded.append({**chunk, "embedding": vector})
                time.sleep(1.0)
            except Exception as e:
                errors.append({"chunk_id": chunk["chunk_id"], "error": str(e)})
                print(f"\n  [WARN] chunk #{chunk['chunk_id']}: {e}")

        out_file = embed_dir / chunk_file.name
        out_file.write_text(json.dumps(embedded, ensure_ascii=False, indent=2), encoding="utf-8")

        total_ok += len(embedded)
        total_err += len(errors)
        print(f"  성공: {len(embedded)} / 실패: {len(errors)}")
        if embedded:
            print(f"  벡터 차원: {len(embedded[0]['embedding'])}")
        print(f"  저장 → {out_file}")

    print(f"\n[완료] 전체 성공: {total_ok} / 실패: {total_err}")
    print(f"임베딩 결과: {embed_dir}")


if __name__ == "__main__":
    main()
